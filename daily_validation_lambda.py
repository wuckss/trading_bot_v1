#!/usr/bin/env python3
"""
Daily Trade Validation & Narrowing Lambda Function (AM & PM)

This AWS Lambda function runs twice daily (AM premarket & PM close) to validate
and narrow down trades from both OpenAI and Claude APIs.
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL CONFIGURATION - Update these when newer models are released
OPENAI_MODEL = "gpt-4o"  # Latest GPT-4 model
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Latest Claude 3.5 Sonnet
ENABLE_DEEP_RESEARCH = False  # Standard mode for daily validation (change to True if needed)

# .env.example file path
ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), '.env.example')

# Daily validation prompt
DAILY_VALIDATION_PROMPT = """Title: Daily Trade Validation & Narrowing — Micro/Mid-Cap Desk

You are the daily risk review + tactical trading desk. Your goal is to refine yesterday's "A-book" list of 6 names into the TOP 3 actionable trades for today's session based on **current price, fresh news, and whether the original thesis still holds.**

Inputs:
- Yesterday's A-book table (6 ideas with entries/stops/targets).
- Watchlist tickers (10–15).
- Live data: premarket/open prices, overnight news/filings, unusual volume/flow.
- Time horizon: intraday to 5 days.
- Account size & risk: same as research prompt (use per-trade risk 2% of equity max).

Your Process (Daily Workflow):
1. **Update Prices & Technicals:** Pull current price, % change from entry, check if stop triggered overnight, or if setup broken (below key MA/VWAP).
2. **Catalyst & News Check:** Scan fresh 8-Ks, press releases, analyst notes, borrow rate changes, social sentiment spikes, sector news. Flag anything thesis-breaking (guidance cuts, dilutions, downgrade swarms).
3. **Score Update:** Re-score each name quickly (0–5) on: 
   - Setup Integrity (technical structure still valid?)
   - Catalyst Proximity (still in play within 5d?)
   - Volume/Liquidity (confirming or fading?)
   - Risk/Reward (still ≥2:1?)
4. **Cull & Focus:** Drop any that fail >1 key condition; pick the strongest 3 by updated score.
5. **Tactical Plan:** For each of the Top 3, produce:
   - Ticker, current price vs yesterday's entry
   - Thesis reaffirmation or adjustment (≤3 lines)
   - Adjusted entry range / add-on trigger
   - Adjusted stop-loss (tighten if needed)
   - Targets T1/T2 (move up/down if structure changed)
   - Expected 5-day move % and confidence rating
   - Execution note: market open vs midday entry, watch for liquidity windows
6. **Risk Overview:** Summarize portfolio heat if all 3 triggered; flag correlation or sector concentration risk.

Mandatory Output Format:
- Compact 3-row table: Ticker | Curr Px | Entry | Stop | Targets | Thesis Update | Confidence (1-5)
- Bullet list of quick watchlist notes: any names upgraded/downgraded to watch closely or dropped entirely.
- Short 3-line Risk Summary: aggregate exposure, sector skew, key event dates to watch.

Tone & Style:
- Be decisive, actionable, and concise — this is for a live trading desk.
- Use hard numbers (no "around" or "roughly") for entry/stop/targets.
- No disclaimers — output trade plan only.

Daily cadence:
- Run premarket (8am) and optionally 3:30pm (EOD) to plan overnight holds or cuts.

If no A-book idea remains valid, build new shortlist from Watchlist, using same quick scoring & validation process."""


class DailyValidationProcessor:
    """Handles daily trade validation and Google Sheets population"""
    
    def __init__(self, session_type: str = "AM"):
        self.spreadsheet_id = "1zBze8TrvAwYi4zH8qbL-pyMWnJ7gQKGRji8rL1VmKoM"
        self.session_type = session_type  # "AM" or "PM"
        self.sheet_name = f"Daily_{session_type}"  # "Daily_AM" or "Daily_PM"
        
        # Initialize API clients
        self._init_google_sheets()
        self._init_openai()
        self._init_claude()
    
    def _load_env_file(self) -> Dict[str, str]:
        """Load environment variables from .env.example file"""
        env_vars = {}
        try:
            if os.path.exists(ENV_FILE_PATH):
                with open(ENV_FILE_PATH, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            env_vars[key] = value
            return env_vars
        except Exception as e:
            logger.warning(f"Failed to load .env.example file: {e}")
            return {}
    
    def _get_api_key(self, key_name: str) -> str:
        """Get API key from .env.example file or environment variable"""
        # First try loading from .env.example file
        env_vars = self._load_env_file()
        env_key = env_vars.get(key_name, "").strip()
        
        # Check if it's not a placeholder
        if env_key and not env_key.startswith('sk-your-') and not env_key.startswith('sk-ant-your-') and not 'your-project-id' in env_key:
            return env_key
        
        # Fall back to environment variable
        env_key = os.environ.get(key_name, "").strip()
        if env_key:
            return env_key
            
        raise ValueError(f"API key {key_name} not found in .env.example file or environment variables. Please update .env.example with your actual API keys.")
    
    def _init_google_sheets(self):
        """Initialize Google Sheets API client"""
        try:
            # Get service account credentials from hardcoded config or environment
            google_json = self._get_api_key('GOOGLE_SERVICE_ACCOUNT_JSON')
            service_account_info = json.loads(google_json)
            credentials = Credentials.from_service_account_info(service_account_info)
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            logger.info("Google Sheets API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets API: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI API client"""
        try:
            api_key = self._get_api_key('OPENAI_API_KEY')
            openai.api_key = api_key
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API: {e}")
            raise
    
    def _init_claude(self):
        """Initialize Claude API client"""
        try:
            api_key = self._get_api_key('CLAUDE_API_KEY')
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Claude API: {e}")
            raise
    
    def get_previous_research_data(self) -> str:
        """Fetch previous A-book and watchlist data from Weekly_Research sheet"""
        try:
            logger.info("Fetching previous research data from Weekly_Research sheet")
            
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range="Weekly_Research!A:U"
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.warning("No data found in Weekly_Research sheet")
                return "No previous research data available."
            
            # Convert to readable format for AI processing
            data_summary = "PREVIOUS RESEARCH DATA:\n\n"
            
            # Add headers if available
            if len(values) > 0:
                headers = values[0]
                data_summary += f"Headers: {' | '.join(headers)}\n\n"
            
            # Add recent data rows (last 20 rows to avoid token limits)
            recent_rows = values[-20:] if len(values) > 20 else values[1:]
            data_summary += "Recent A-book and Research Data:\n"
            
            for i, row in enumerate(recent_rows, 1):
                # Pad row to match header length
                padded_row = row + [''] * (len(headers) - len(row)) if len(values) > 0 else row
                data_summary += f"Row {i}: {' | '.join(padded_row[:10])}...\n"  # First 10 columns
            
            return data_summary
            
        except Exception as e:
            logger.error(f"Failed to fetch previous research data: {e}")
            return f"Error fetching previous data: {str(e)}"
    
    def clear_sheet_except_header(self) -> bool:
        """Clear the daily sheet except the header row"""
        try:
            # First, get the current sheet data to determine range
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheet_name}!A:Z"
            ).execute()
            
            values = result.get('values', [])
            if len(values) <= 1:
                logger.info("Sheet has only header row or is empty, nothing to clear")
                return True
            
            # Clear everything except the header (row 1)
            clear_range = f"{self.sheet_name}!A2:Z{len(values)}"
            self.sheets_service.spreadsheets().values().clear(
                spreadsheetId=self.spreadsheet_id,
                range=clear_range
            ).execute()
            
            logger.info(f"Cleared sheet range: {clear_range}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear sheet: {e}")
            return False
    
    def get_openai_validation(self, previous_data: str) -> Optional[str]:
        """Generate validation using OpenAI's model"""
        try:
            logger.info(f"Generating OpenAI validation using model: {OPENAI_MODEL}")
            
            # Prepare the validation prompt
            session_context = f"Session Type: {self.session_type} ({'Premarket Analysis' if self.session_type == 'AM' else 'End of Day Review'})"
            
            validation_prompt = f"""
            {session_context}
            Current date/time: {datetime.now(timezone.utc).isoformat()}
            
            {DAILY_VALIDATION_PROMPT}
            
            {previous_data}
            
            VALIDATION REQUIREMENTS:
            - Focus on the TOP 3 actionable trades for today
            - Update prices and check if stops were triggered
            - Validate thesis integrity with current market conditions
            - Provide specific entry/exit levels (no ranges)
            - Include risk summary and portfolio heat assessment
            """
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional trading desk analyst providing daily trade validation and risk assessment. Be concise and actionable."
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.2  # Low temperature for consistent analysis
            )
            
            validation_content = response.choices[0].message.content
            logger.info("OpenAI validation generated successfully")
            return validation_content
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI validation: {e}")
            return None
    
    def get_claude_validation(self, previous_data: str) -> Optional[str]:
        """Generate validation using Claude's model"""
        try:
            logger.info(f"Generating Claude validation using model: {CLAUDE_MODEL}")
            
            session_context = f"Session Type: {self.session_type} ({'Premarket Analysis' if self.session_type == 'AM' else 'End of Day Review'})"
            
            validation_prompt = f"""
            {session_context}
            Current date/time: {datetime.now(timezone.utc).isoformat()}
            
            {DAILY_VALIDATION_PROMPT}
            
            {previous_data}
            
            VALIDATION REQUIREMENTS:
            - Focus on the TOP 3 actionable trades for today
            - Update prices and check if stops were triggered  
            - Validate thesis integrity with current market conditions
            - Provide specific entry/exit levels (no ranges)
            - Include risk summary and portfolio heat assessment
            """
            
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                temperature=0.2,  # Low temperature for consistent analysis
                system="You are an experienced trading desk analyst providing daily trade validation. Focus on actionable insights and risk management.",
                messages=[
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ]
            )
            
            validation_content = response.content[0].text
            logger.info("Claude validation generated successfully")
            return validation_content
            
        except Exception as e:
            logger.error(f"Failed to generate Claude validation: {e}")
            return None
    
    def parse_validation_to_rows(self, validation_content: str, source: str) -> List[List[str]]:
        """Parse AI validation content into spreadsheet rows"""
        rows = []
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Add source identifier and session type
        rows.append([f"=== {source.upper()} {self.session_type} VALIDATION - {timestamp} ==="])
        rows.append([])  # Empty row for spacing
        
        # Split content into lines and process
        lines = validation_content.split('\n')
        current_section = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_section:
                    # Process accumulated section
                    for section_line in current_section:
                        rows.append([section_line])
                    current_section = []
                    rows.append([])  # Add spacing
            else:
                current_section.append(line)
        
        # Handle any remaining content
        if current_section:
            for section_line in current_section:
                rows.append([section_line])
        
        rows.append([])  # Final spacing
        rows.append(["=" * 50])  # Separator
        rows.append([])
        
        return rows
    
    def write_to_sheet(self, openai_content: Optional[str], claude_content: Optional[str]) -> bool:
        """Write both AI validation results to Google Sheets"""
        try:
            all_rows = []
            
            # Create headers for daily validation
            headers = [
                f"{self.session_type} Validation Content", "Additional Notes", "Timestamp", 
                "AI Source", "Session Type", "Status"
            ]
            
            # Check if headers exist, if not create them
            try:
                result = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{self.sheet_name}!A1:F1"
                ).execute()
                
                existing_headers = result.get('values', [[]])
                if not existing_headers or not existing_headers[0]:
                    # Write headers
                    self.sheets_service.spreadsheets().values().update(
                        spreadsheetId=self.spreadsheet_id,
                        range=f"{self.sheet_name}!A1:F1",
                        valueInputOption='RAW',
                        body={'values': [headers]}
                    ).execute()
                    logger.info("Headers created successfully")
            except Exception as e:
                logger.warning(f"Could not check/create headers: {e}")
            
            # Add OpenAI content if available
            if openai_content:
                openai_rows = self.parse_validation_to_rows(openai_content, "OpenAI")
                all_rows.extend(openai_rows)
            
            # Add Claude content if available
            if claude_content:
                claude_rows = self.parse_validation_to_rows(claude_content, "Claude")
                all_rows.extend(claude_rows)
            
            if not all_rows:
                logger.error("No content to write to sheet")
                return False
            
            # Write to sheet starting from row 2 (after header)
            range_name = f"{self.sheet_name}!A2:A{len(all_rows) + 1}"
            
            body = {
                'values': all_rows
            }
            
            result = self.sheets_service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Successfully wrote {len(all_rows)} rows to {self.sheet_name} sheet")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to sheet: {e}")
            return False
    
    def process_daily_validation(self) -> Dict[str, Any]:
        """Main processing function for daily validation"""
        results = {
            'success': False,
            'session_type': self.session_type,
            'openai_success': False,
            'claude_success': False,
            'sheet_cleared': False,
            'sheet_updated': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        try:
            # Step 1: Get previous research data
            previous_data = self.get_previous_research_data()
            
            # Step 2: Clear the sheet except header
            results['sheet_cleared'] = self.clear_sheet_except_header()
            
            # Step 3: Generate validation from both AIs
            openai_content = None
            claude_content = None
            
            try:
                openai_content = self.get_openai_validation(previous_data)
                results['openai_success'] = openai_content is not None
            except Exception as e:
                results['errors'].append(f"OpenAI error: {str(e)}")
            
            try:
                claude_content = self.get_claude_validation(previous_data)
                results['claude_success'] = claude_content is not None
            except Exception as e:
                results['errors'].append(f"Claude error: {str(e)}")
            
            # Step 4: Write results to sheet
            if openai_content or claude_content:
                results['sheet_updated'] = self.write_to_sheet(openai_content, claude_content)
            
            # Overall success if at least one AI succeeded and sheet was updated
            results['success'] = (results['openai_success'] or results['claude_success']) and results['sheet_updated']
            
        except Exception as e:
            logger.error(f"Critical error in daily validation processing: {e}")
            results['errors'].append(f"Critical error: {str(e)}")
        
        return results


def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Daily validation Lambda function started")
    
    try:
        # Determine session type from event or default to AM
        session_type = event.get('session_type', 'AM')  # 'AM' or 'PM'
        
        # Initialize processor
        processor = DailyValidationProcessor(session_type=session_type)
        
        # Process daily validation
        results = processor.process_daily_validation()
        
        # Return results
        return {
            'statusCode': 200 if results['success'] else 500,
            'body': json.dumps(results, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }


if __name__ == "__main__":
    # For local testing
    import sys
    
    # Allow command line session type argument
    session_type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ['AM', 'PM'] else 'AM'
    
    # Mock event and context for testing
    test_event = {'session_type': session_type}
    test_context = type('Context', (), {
        'function_name': 'daily_validation_lambda',
        'function_version': '1',
        'invoked_function_arn': f'arn:aws:lambda:us-east-1:123456789:function:daily_validation_lambda_{session_type.lower()}',
        'memory_limit_in_mb': '512',
        'remaining_time_in_millis': lambda: 30000
    })()
    
    print(f"Running {session_type} session validation...")
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2))