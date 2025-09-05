#!/usr/bin/env python3
"""
Daily Trade Validation & Narrowing Lambda Function (AM & PM)

This AWS Lambda function runs twice daily (AM premarket & PM close) to validate
and narrow down trades from both OpenAI and Claude APIs.
"""

import json
import os
import logging
import time
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import gspread
from google.oauth2.service_account import Credentials
import openai
import anthropic
import pandas_market_calendars as mcal
import pytz
import requests

# For local development
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists (local development only)
except ImportError:
    pass  # python-dotenv not available in Lambda runtime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-opus-4-1')
ENABLE_DEEP_RESEARCH = os.getenv('ENABLE_DEEP_RESEARCH', 'false').lower() == 'true'

# Google Sheets configuration
DAILY_SHEET_ID = os.getenv('GOOGLE_SHEETS_DAILY_ID', '1zBze8TrvAwYi4zH8qbL-pyMWnJ7gQKGRji8rL1VmKoM')
DAILY_AM_SHEET_NAME = os.getenv('DAILY_AM_SHEET_NAME', 'Daily_AM')
DAILY_PM_SHEET_NAME = os.getenv('DAILY_PM_SHEET_NAME', 'Daily_PM')

# Market and timezone configuration
US_EASTERN = pytz.timezone('US/Eastern')
NYSE_CALENDAR = mcal.get_calendar('NYSE')

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

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
        self.spreadsheet_id = DAILY_SHEET_ID
        self.session_type = session_type  # "AM" or "PM"
        self.sheet_name = DAILY_AM_SHEET_NAME if session_type == "AM" else DAILY_PM_SHEET_NAME
        self.run_id = self._generate_run_id()
        
        # Initialize API clients
        self._init_google_sheets()
        self._init_openai()
        self._init_anthropic()
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID for this execution"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"DV_{self.session_type}_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _get_env_var(self, key: str, required: bool = True) -> str:
        """Get environment variable with proper error handling"""
        value = os.getenv(key)
        if not value and required:
            raise ValueError(f"Required environment variable {key} not found")
        return value or ""
    
    def _generate_idempotency_key(self) -> str:
        """Generate idempotency key based on date + session + timestamp bucket"""
        now_et = datetime.now(US_EASTERN)
        
        # Create timestamp bucket based on session type
        if self.session_type == "AM":
            # AM session: 08:00 ET bucket
            bucket_time = "08ET"
        else:
            # PM session: 15:30 ET bucket  
            bucket_time = "1530ET"
        
        # Create the bucket identifier
        bucket_id = f"daily_{self.session_type}_{now_et.date()}_{bucket_time}"
        
        # Generate hash for idempotency
        idempotency_key = hashlib.md5(bucket_id.encode()).hexdigest()[:12]
        
        logger.info(f"Generated idempotency key: {idempotency_key} for bucket: {bucket_id}")
        return idempotency_key
    
    def _is_trading_day(self) -> bool:
        """Check if today is a trading day using NYSE calendar"""
        try:
            now_et = datetime.now(US_EASTERN)
            today = now_et.date()
            
            # Get trading sessions for today
            trading_days = NYSE_CALENDAR.sessions_in_range(today, today)
            
            is_trading_day = len(trading_days) > 0
            logger.info(f"Trading day check: {today} is {'a' if is_trading_day else 'not a'} trading day")
            
            if not is_trading_day:
                logger.info("Not a trading day - skipping validation")
                
            return is_trading_day
            
        except Exception as e:
            logger.warning(f"Failed to check trading calendar, proceeding anyway: {e}")
            return True  # Fail open - run validation if calendar check fails
    
    def _is_valid_session_time(self) -> bool:
        """Check if current time is appropriate for the session type"""
        try:
            now_et = datetime.now(US_EASTERN)
            current_time = now_et.time()
            
            if self.session_type == "AM":
                # AM session: 06:00 - 09:30 ET (before market open)
                start_time = datetime.strptime("06:00", "%H:%M").time()
                end_time = datetime.strptime("09:30", "%H:%M").time()
                valid = start_time <= current_time <= end_time
                session_window = "06:00-09:30 ET (premarket)"
            else:
                # PM session: 15:00 - 18:00 ET (after market close)
                start_time = datetime.strptime("15:00", "%H:%M").time()
                end_time = datetime.strptime("18:00", "%H:%M").time()
                valid = start_time <= current_time <= end_time
                session_window = "15:00-18:00 ET (post-market)"
            
            logger.info(f"Session time check: Current time {current_time.strftime('%H:%M')} ET, "
                       f"valid window {session_window}: {'✓' if valid else '✗'}")
            
            return valid
            
        except Exception as e:
            logger.warning(f"Failed to check session time, proceeding anyway: {e}")
            return True  # Fail open
    
    def _check_idempotency(self) -> bool:
        """Check if this validation has already been generated (idempotency check)"""
        try:
            idempotency_key = self._generate_idempotency_key()
            
            # For daily validation, we only use column A (single column format)
            # Check if this key or run ID exists in the sheet
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheet_name}!A:A"
            ).execute()
            
            values = result.get('values', [])
            
            # Look for the idempotency key or run ID in existing data
            for i, row in enumerate(values):
                if not row or not row[0]:
                    continue
                    
                row_content = str(row[0])
                # Check for idempotency key or run ID in the content
                if (idempotency_key in row_content or 
                    self.run_id in row_content or
                    f"ID:{idempotency_key}" in row_content):
                    logger.info(f"Idempotency check: Validation already exists for key {idempotency_key} or Run ID {self.run_id} (row {i+1})")
                    return False  # Already exists, skip
            
            logger.info(f"Idempotency check: No existing validation for key {idempotency_key} or Run ID {self.run_id}, proceeding")
            return True  # Doesn't exist, proceed
            
        except Exception as e:
            logger.warning(f"Idempotency check failed, proceeding anyway: {e}")
            return True  # Fail open
    
    def _init_google_sheets(self):
        """Initialize Google Sheets API client"""
        try:
            # Get base64 encoded service account JSON
            google_json_b64 = self._get_env_var('GOOGLE_SERVICE_ACCOUNT_JSON_BASE64')
            google_json = base64.b64decode(google_json_b64).decode('utf-8')
            service_account_info = json.loads(google_json)
            credentials = Credentials.from_service_account_info(service_account_info)
            credentials = credentials.with_scopes(["https://www.googleapis.com/auth/spreadsheets"])
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            logger.info("Google Sheets API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets API: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI API client"""
        try:
            # OpenAI client automatically reads OPENAI_API_KEY from environment
            self.openai_client = openai.OpenAI()
            logger.info(f"OpenAI API initialized successfully with model: {OPENAI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API: {e}")
            raise
    
    def _init_anthropic(self):
        """Initialize Anthropic API client"""
        try:
            api_key = self._get_env_var('ANTHROPIC_API_KEY')
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Anthropic API initialized successfully with model: {ANTHROPIC_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic API: {e}")
            raise
    
    def _send_telegram_notification(self, message: str, session_type: str) -> bool:
        """Send Telegram notification with validation results"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured, skipping notification")
            return False
            
        try:
            # Format message for Telegram
            formatted_message = f"🔔 *Daily {session_type} Validation Complete*\n\n{message}"
            
            # Telegram API URL
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            
            # Prepare payload
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': formatted_message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            # Send with retry logic
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        logger.info(f"Telegram notification sent successfully for {session_type} session")
                        return True
                    else:
                        logger.warning(f"Telegram API returned status {response.status_code}: {response.text}")
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Telegram request timeout on attempt {attempt + 1}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Telegram request failed on attempt {attempt + 1}: {e}")
                
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            logger.error("Failed to send Telegram notification after 3 attempts")
            return False
            
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")
            return False
    
    def _format_telegram_message(self, combined_content: str, session_type: str) -> str:
        """Format the validation content for Telegram"""
        try:
            # Truncate if too long (Telegram has 4096 char limit)
            if len(combined_content) > 3500:
                combined_content = combined_content[:3500] + "...\n\n[Content truncated for Telegram]"
            
            # Add session info and timestamp
            now_et = datetime.now(US_EASTERN)
            timestamp = now_et.strftime("%Y-%m-%d %H:%M EST")
            
            formatted = f"📊 *{session_type.upper()} Session Results*\n"
            formatted += f"⏰ {timestamp}\n"
            formatted += f"🎯 Run ID: `{self.run_id}`\n\n"
            formatted += "📋 *Validation Summary:*\n"
            formatted += f"```\n{combined_content}\n```"
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting Telegram message: {e}")
            # Fallback to simple format
            return f"{session_type.upper()} Validation Complete\n\n{combined_content[:3000]}"
    
    def _combine_validation_content(self, openai_content: Optional[str], claude_content: Optional[str]) -> str:
        """Combine OpenAI and Claude validation content for Telegram"""
        combined = ""
        
        if openai_content:
            combined += f"🤖 OpenAI Analysis:\n{openai_content[:1500]}\n\n"
        
        if claude_content:
            combined += f"🧠 Claude Analysis:\n{claude_content[:1500]}\n\n"
        
        if not combined:
            combined = "No validation content generated."
            
        return combined.strip()
    
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
            Run ID: {self.run_id}
            
            {DAILY_VALIDATION_PROMPT}
            
            {previous_data}
            
            VALIDATION REQUIREMENTS:
            - Focus on the TOP 3 actionable trades for today
            - Update prices and check if stops were triggered
            - Validate thesis integrity with current market conditions
            - Provide specific entry/exit levels (no ranges)
            - Include risk summary and portfolio heat assessment
            """
            
            # Retry logic with exponential backoff
            for attempt in range(3):
                try:
                    response = self.openai_client.chat.completions.create(
                        model=OPENAI_MODEL,  # gpt-5 or gpt-5-mini
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a precise, concise trading assistant providing daily trade validation and risk assessment."
                            },
                            {
                                "role": "user",
                                "content": validation_prompt
                            }
                        ],
                        max_tokens=4096,  # Increased for daily validation
                        temperature=0.2,  # Precise temperature
                        timeout=60  # 1 minute timeout
                    )
                    
                    validation_content = response.choices[0].message.content
                    logger.info(f"OpenAI validation generated successfully on attempt {attempt + 1}")
                    return validation_content
                    
                except Exception as e:
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise e
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI validation: {e}")
            return None
    
    def get_anthropic_validation(self, previous_data: str) -> Optional[str]:
        """Generate validation using Anthropic's Claude model"""
        try:
            logger.info(f"Generating Claude validation using model: {CLAUDE_MODEL}")
            
            session_context = f"Session Type: {self.session_type} ({'Premarket Analysis' if self.session_type == 'AM' else 'End of Day Review'})"
            
            validation_prompt = f"""
            {session_context}
            Current date/time: {datetime.now(timezone.utc).isoformat()}
            Run ID: {self.run_id}
            
            {DAILY_VALIDATION_PROMPT}
            
            {previous_data}
            
            VALIDATION REQUIREMENTS:
            - Focus on the TOP 3 actionable trades for today
            - Update prices and check if stops were triggered  
            - Validate thesis integrity with current market conditions
            - Provide specific entry/exit levels (no ranges)
            - Include risk summary and portfolio heat assessment
            """
            
            # Retry logic with exponential backoff
            for attempt in range(3):
                try:
                    response = self.anthropic_client.messages.create(
                        model=ANTHROPIC_MODEL,  # claude-opus-4-1 or pinned version
                        max_tokens=4096,  # Increased for daily validation
                        temperature=0.2,  # Precise temperature
                        system="You are a precise, risk-aware trading researcher providing daily trade validation. Focus on actionable insights and risk management.",
                        messages=[
                            {
                                "role": "user",
                                "content": validation_prompt
                            }
                        ],
                        timeout=60  # 1 minute timeout
                    )
                    
                    validation_content = response.content[0].text
                    logger.info(f"Claude validation generated successfully on attempt {attempt + 1}")
                    return validation_content
                    
                except Exception as e:
                    logger.warning(f"Claude API attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise e
            
        except Exception as e:
            logger.error(f"Failed to generate Claude validation after retries: {e}")
            return None
    
    def parse_validation_to_rows(self, validation_content: str, source: str) -> List[List[str]]:
        """Parse AI validation content into spreadsheet rows"""
        rows = []
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        idempotency_key = self._generate_idempotency_key()
        
        # Add source identifier and session type with idempotency key
        rows.append([f"=== {source.upper()} {self.session_type} VALIDATION - {timestamp} - ID:{idempotency_key} ==="])
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
            
            # Create headers for daily validation - single column for content
            headers = ["Content"]
            
            # Check if headers exist, if not create them
            try:
                result = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{self.sheet_name}!A1:A1"
                ).execute()
                
                existing_headers = result.get('values', [[]])
                if not existing_headers or not existing_headers[0]:
                    # Write headers
                    self.sheets_service.spreadsheets().values().update(
                        spreadsheetId=self.spreadsheet_id,
                        range=f"{self.sheet_name}!A1:A1",
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
            
            # Write to sheet starting from row 2 (after header) - use anchor only for safety
            range_name = f"{self.sheet_name}!A2"
            
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
            'skipped': False,
            'skip_reason': '',
            'session_type': self.session_type,
            'openai_success': False,
            'claude_success': False,
            'sheet_cleared': False,
            'sheet_updated': False,
            'is_trading_day': False,
            'valid_session_time': False,
            'idempotency_check': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        try:
            # Step 1: Trading day check
            results['is_trading_day'] = self._is_trading_day()
            if not results['is_trading_day']:
                results['skipped'] = True
                results['skip_reason'] = 'Not a trading day (market holiday/weekend)'
                return results
            
            # Step 2: Session time check
            results['valid_session_time'] = self._is_valid_session_time()
            if not results['valid_session_time']:
                results['skipped'] = True
                results['skip_reason'] = f'Outside valid {self.session_type} session time window'
                return results
            
            # Step 3: Idempotency check
            results['idempotency_check'] = self._check_idempotency()
            if not results['idempotency_check']:
                results['skipped'] = True
                results['skip_reason'] = f'Validation already generated for this {self.session_type} session (idempotency)'
                return results
            
            # Step 4: Get previous research data
            previous_data = self.get_previous_research_data()
            
            # Step 5: Clear the sheet except header
            results['sheet_cleared'] = self.clear_sheet_except_header()
            
            # Step 6: Generate validation from both AIs
            openai_content = None
            claude_content = None
            
            try:
                openai_content = self.get_openai_validation(previous_data)
                results['openai_success'] = openai_content is not None
            except Exception as e:
                results['errors'].append(f"OpenAI error: {str(e)}")
            
            try:
                claude_content = self.get_anthropic_validation(previous_data)
                results['claude_success'] = claude_content is not None
            except Exception as e:
                results['errors'].append(f"Claude error: {str(e)}")
            
            # Step 7: Write results to sheet
            if openai_content or claude_content:
                results['sheet_updated'] = self.write_to_sheet(openai_content, claude_content)
                
                # Step 8: Send Telegram notification if successful
                if results['sheet_updated']:
                    combined_content = self._combine_validation_content(openai_content, claude_content)
                    telegram_message = self._format_telegram_message(combined_content, self.session_type)
                    results['telegram_sent'] = self._send_telegram_notification(telegram_message, self.session_type)
            
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
