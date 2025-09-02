#!/usr/bin/env python3
"""
Weekly AI Trading Research Lambda Function

This AWS Lambda function runs weekly to populate Google Sheets with trading research
from both OpenAI and Claude APIs using the latest models.
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import boto3
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL CONFIGURATION - Update these when newer models are released
OPENAI_MODEL = "gpt-4o"  # Latest GPT-4 model with research capabilities
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Latest Claude 3.5 Sonnet
ENABLE_DEEP_RESEARCH = True  # Enable enhanced research mode for both models

# .env.example file path
ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), '.env.example')

# Trading research prompt
TRADING_RESEARCH_PROMPT = """Title: 5-Day Alpha Sprint — Micro/Mid-Cap Trading Desk (High Risk Small Account)

CRITICAL FORMATTING REQUIREMENT: Your final output MUST include a structured table with this EXACT format for each trade idea:

TICKER | CURRENT_PRICE | MARKET_CAP | AVG_VOLUME | SETUP_TYPE | CATALYST_DATE | THESIS | ENTRY | STOP | T1 | T2 | T3 | SIZE% | RR | 5D_MOVE% | CONF | RISKS | OPTIONS | EXECUTION

Example:
ABCD | $24.85 | $2.5B | $45M | Breakout | 2024-01-15 | Strong earnings momentum with technical breakout above $25 resistance | $25.50 | $23.00 | $28 | $31 | $35 | 3% | 2.8:1 | +15% | 4 | Earnings miss risk | Feb 30C @ $2.50 | Watch opening gap

You are a coordinated Wall Street trading firm operating as a multi-specialist team. Your single goal: maximize P&L over the next 5 trading days with disciplined risk on a small, high-risk account. Work collaboratively, challenge each other, and deliver only trade-ready ideas with explicit entries/exits and risk controls.

Assumptions (editable defaults):
- Time horizon: next 5 trading days (T+0 to T+4), timezone: America/Toronto.
- Account size: $25,000 (scale all sizing as % of equity).
- Max concurrent positions: 6; max portfolio risk-at-stop: 10% of equity.
- Asset universe: U.S.-listed micro/mid caps, market cap $150M–$8B, price ≥ $2.
- Liquidity floor: Avg daily $ volume ≥ $10M; avoid OTC/pink, active delisting, or <$2.
- Options allowed but use defined risk (debit spreads/calls) only if spreads are tight and OI supports entry/exit.

Team & Responsibilities:
1) Lead PM: Set constraints, enforce risk, final selection.
2) Catalyst Hunter: Surface near-term (≤5 days) catalysts—earnings, guidance, FDA meetings/PRs, conference presentations, investor days, contract awards, court rulings, financings/lockups, index adds/deletes, short squeeze triggers (borrow rate spikes, DTC↑).
3) Quant Screener: Filter universe by momentum (RSI, 20/50/200MA posture), 5-day realized vs implied move, unusual volume, short interest %, days-to-cover, put/call, insider/institutional flow; flag valuation asymmetry (EV/Sales vs growth).
4) Technical Strategist: Identify A+ setups for 5-day swing: breakouts from multi-week bases, earnings gaps with tight flags, prior resistance turned support, VWAP reclaim, trend alignment across timeframes; specify entry zone, invalidation, R:R ≥ 2:1.
5) Options Architect: Where liquid, propose defined-risk structures aligned to catalyst date (OTM calls or call spreads, ≤1–2 weeks expiry), note breakeven, max loss, target exits.
6) Risk & Compliance: Enforce guardrails (position sizing, max loss per trade ≤2% of equity, avoid illiquid traps), list key risks (financing overhang, warrants, binary biotech risk).
7) News & Sentiment Sentinel: Scan real-time news, 8-K/PRs, social sentiment velocity, analyst notes; tag red/green flags and misinformation risk.
8) Execution Trader: Provide bracket order plan (entry ladder, stop, targets), expected slippage, pre-market/after-hours considerations.
9) Post-Mortem Analyst: Define success/failure checkpoints at EOD each day; specify rule-based adjustments (trail stops, scale-outs on +1R, cut losers quickly).

Scoring & Selection (numerical, 0–10 each; weight in parentheses):
- Catalyst Imminence (0.30): binary/timed within 5 days with clear path to re-pricing.
- Momentum/Setup Quality (0.20): structure, volume, higher-timeframe alignment.
- Liquidity/Tradability (0.15): $ volume, spread (shares or options), borrow availability.
- Asymmetry/Valuation (0.15): plausible 5-day upside vs downside; peer comps.
- Short-Interest & Squeeze Factors (0.10): SI% float, DTC, borrow fees.
- Sentiment/Flow (0.10): news tone, social acceleration, unusual option flow.

Compute Total Score = Σ(weight × subscore). Return only top candidates with Total Score ≥ 3.6/5 and R:R ≥ 2:1.

Mandatory Output — Actionable Trade Sheet (Top 6 "A-book" ideas + 10-15 name watchlist):
For each A-book idea, present a compact row with:
- Ticker | Market Cap | Avg $Vol | Setup Type
- Near-Term Catalyst & Date/Window (exact timing if known)
- Thesis (≤3 crisp lines)
- Entry Zone (price) | Initial Stop (price/%) | Targets T1/T2/T3
- Position Size % (of equity) and R:R
- Expected 5-Day Move % (basis: IV/analog events/ATR) and Confidence (1–10)
- Risks/Invalidation (specific)
- Options Alt (if viable): structure, expiration (≤2w), debit, BE, max loss, exits
- Execution Notes (liquidity windows, open/close risk, news tape to watch)

Watchlist (second tier):
- 10–15 tickers with 1-line thesis + catalyst timing + trigger level; no entries until trigger fires.

Process & Cadence (deliver all in one response):
- Universe Build: Apply screens/filters; list top ~60; cut to ~20 via scoring; pick top 6.
- Premarket Playbook: note gap risks, fresh PRs/8-Ks, borrow changes, halts.
- Risk Plan: portfolio heat map at entry and at stop; correlation check; event-time exposure limits.
- Day-By-Day Plan (D0–D4): what would cause adds, trims, or exits; trailing logic; news checkpoints.

Formatting:
- Start with a 6-row "A-book" table followed by a compact watchlist table.
- Be specific with numbers (entries/stops/targets), not ranges like "around".
- If data is missing, make conservative, clearly-labeled assumptions and proceed.
- No disclaimers, no education—return the trade plan only.

User Inputs (if provided, override defaults):
{ACCOUNT_SIZE=$25,000, MAX_POSITIONS=6, MAX_RISK_AT_STOP=10%, PER_TRADE_RISK=2%, MCAP_MIN=$150M, MCAP_MAX=$8B, MIN_PRICE=$2, MIN_DOLLAR_VOL=$10M}

Deliverable: One response containing (1) A-book table, (2) Watchlist table, (3) Premarket notes, (4) Risk heat map summary, (5) Day-by-day management rules."""


class WeeklyResearchProcessor:
    """Handles weekly trading research generation and Google Sheets population"""
    
    def __init__(self):
        self.spreadsheet_id = "1zBze8TrvAwYi4zH8qbL-pyMWnJ7gQKGRji8rL1VmKoM"
        self.sheet_name = "Weekly_Research"
        
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
    
    def clear_sheet_except_header(self) -> bool:
        """Clear the Weekly_Research sheet except the header row"""
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
    
    def get_openai_research(self) -> Optional[str]:
        """Generate research using OpenAI's latest model with deep research mode"""
        try:
            logger.info(f"Generating OpenAI research using model: {OPENAI_MODEL}")
            
            # Prepare the research prompt with enhanced instructions for deep research
            enhanced_prompt = f"""
            DEEP RESEARCH MODE ENABLED: Use all available knowledge, analysis capabilities, and reasoning to provide the most comprehensive trading research possible.
            
            Current date/time: {datetime.now(timezone.utc).isoformat()}
            
            {TRADING_RESEARCH_PROMPT}
            
            RESEARCH ENHANCEMENT INSTRUCTIONS:
            - Analyze current market conditions and sector trends
            - Consider macroeconomic factors affecting micro/mid-caps
            - Evaluate earnings calendar and upcoming catalysts
            - Assess technical patterns and momentum indicators
            - Review recent news, analyst coverage, and institutional activity
            - Provide detailed risk assessment and position sizing logic
            """
            
            # Use the configured model
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Wall Street trading analyst with deep market knowledge and research capabilities. Provide detailed, actionable trading research."
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.1 if ENABLE_DEEP_RESEARCH else 0.7  # Lower temperature for more focused research
            )
            
            research_content = response.choices[0].message.content
            logger.info("OpenAI research generated successfully")
            return research_content
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI research: {e}")
            return None
    
    def get_claude_research(self) -> Optional[str]:
        """Generate research using Claude's latest model with deep research capabilities"""
        try:
            logger.info(f"Generating Claude research using model: {CLAUDE_MODEL}")
            
            # Prepare enhanced system prompt for deep research mode
            system_prompt = """You are an elite Wall Street trading research analyst with deep market expertise and comprehensive analytical capabilities. When DEEP RESEARCH MODE is enabled, utilize your full knowledge base and analytical reasoning to provide the most thorough, actionable trading research possible.

Focus on:
- Real-time market analysis and sector dynamics  
- Comprehensive catalyst identification and timing
- Advanced technical analysis with multiple timeframes
- Institutional flow and sentiment analysis
- Risk-adjusted position sizing and portfolio construction
- Macroeconomic context and market regime analysis"""

            enhanced_prompt = f"""
            DEEP RESEARCH MODE: {'ENABLED' if ENABLE_DEEP_RESEARCH else 'STANDARD'} 
            
            Current date/time: {datetime.now(timezone.utc).isoformat()}
            
            {TRADING_RESEARCH_PROMPT}
            
            {'ENHANCED RESEARCH REQUIREMENTS:' if ENABLE_DEEP_RESEARCH else ''}
            {'''- Conduct thorough fundamental and technical analysis
            - Analyze earnings estimates, revisions, and guidance trends  
            - Evaluate insider trading, institutional positioning, and short interest
            - Consider options flow, volatility patterns, and market structure
            - Assess sector rotation, relative strength, and correlation factors
            - Provide detailed execution strategies and risk management protocols''' if ENABLE_DEEP_RESEARCH else ''}
            """
            
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8000,
                temperature=0.1 if ENABLE_DEEP_RESEARCH else 0.3,  # Lower temperature for research mode
                system=system_prompt if ENABLE_DEEP_RESEARCH else "You are an experienced trading analyst providing actionable market research.",
                messages=[
                    {
                        "role": "user", 
                        "content": enhanced_prompt
                    }
                ]
            )
            
            research_content = response.content[0].text
            logger.info("Claude research generated successfully")
            return research_content
            
        except Exception as e:
            logger.error(f"Failed to generate Claude research: {e}")
            return None
    
    def parse_research_to_rows(self, research_content: str, source: str) -> List[List[str]]:
        """Parse AI research content into structured spreadsheet rows"""
        rows = []
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Try to extract structured trade data from the response
        trade_rows = self._extract_trade_data(research_content, source, timestamp)
        
        if trade_rows:
            # If we successfully extracted structured data, use it
            rows.extend(trade_rows)
        else:
            # Fallback to free-form text if parsing fails
            rows.append([f"=== {source.upper()} RESEARCH - {timestamp} ==="])
            rows.append([])
            
            lines = research_content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    rows.append([line])
            
            rows.append([])
            rows.append(["=" * 50])
            rows.append([])
        
        return rows
    
    def _extract_trade_data(self, content: str, source: str, timestamp: str) -> List[List[str]]:
        """Extract structured trade data from AI response"""
        import re
        
        trade_rows = []
        
        # Look for A-book table or structured trade ideas
        lines = content.split('\n')
        in_table = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Look for ticker patterns (3-5 uppercase letters)
            ticker_match = re.match(r'^([A-Z]{2,5})\s*[\|\-\:]', line)
            if ticker_match:
                ticker = ticker_match.group(1)
                
                # Try to parse the structured data from this line and following context
                trade_data = self._parse_trade_line(line, lines[i:i+10], source, timestamp)
                if trade_data:
                    trade_rows.append(trade_data)
            
            # Alternative: Look for pipe-separated table rows
            elif '|' in line and not line.startswith('='):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3 and re.match(r'^[A-Z]{2,5}$', parts[0].strip()):
                    # This looks like a table row
                    trade_data = self._parse_pipe_separated_row(parts, source, timestamp)
                    if trade_data:
                        trade_rows.append(trade_data)
        
        return trade_rows
    
    def _parse_trade_line(self, line: str, context_lines: List[str], source: str, timestamp: str) -> Optional[List[str]]:
        """Parse a single trade idea line with context"""
        import re
        
        # Extract ticker
        ticker_match = re.match(r'^([A-Z]{2,5})', line)
        if not ticker_match:
            return None
            
        ticker = ticker_match.group(1)
        
        # Initialize row with empty values for all columns
        row = [''] * 21  # 21 columns total (added Current Price)
        row[0] = ticker  # Ticker
        row[18] = source  # AI Source
        row[19] = timestamp  # Generated Timestamp
        
        # Join the context to search for patterns
        full_context = ' '.join(context_lines[:5])
        
        # Extract current price (first price found)
        price_match = re.search(r'\$(\d+\.?\d*)', full_context)
        if price_match:
            row[1] = f"${price_match.group(1)}"
        
        # Extract market cap
        mcap_match = re.search(r'\$(\d+(?:\.\d+)?[BMK])', full_context)
        if mcap_match:
            row[2] = mcap_match.group(1)
        
        # Extract entry price patterns
        entry_match = re.search(r'Entry[:\s]*\$?(\d+\.?\d*)', full_context, re.IGNORECASE)
        if entry_match:
            row[7] = f"${entry_match.group(1)}"
        
        # Extract stop loss
        stop_match = re.search(r'Stop[:\s]*\$?(\d+\.?\d*)', full_context, re.IGNORECASE)
        if stop_match:
            row[8] = f"${stop_match.group(1)}"
        
        # Extract targets
        target_matches = re.findall(r'T[1-3][:\s]*\$?(\d+\.?\d*)', full_context, re.IGNORECASE)
        for i, target in enumerate(target_matches[:3]):
            row[9 + i] = f"${target}"
        
        # Extract percentage values
        pct_matches = re.findall(r'(\d+(?:\.\d+)?%)', full_context)
        if pct_matches:
            row[12] = pct_matches[0]  # Position size %
        
        # Fill in basic info from line
        remaining_text = re.sub(r'^[A-Z]{2,5}\s*[\|\-\:]\s*', '', line)
        if remaining_text:
            row[6] = remaining_text[:100]  # Thesis (truncated)
        
        return row
    
    def _parse_pipe_separated_row(self, parts: List[str], source: str, timestamp: str) -> Optional[List[str]]:
        """Parse a pipe-separated table row"""
        if len(parts) < 2:
            return None
            
        row = [''] * 21
        
        # Map parts to appropriate columns (shifted by 1 due to Current Price column)
        row[0] = parts[0] if len(parts) > 0 else ''  # Ticker
        row[1] = parts[1] if len(parts) > 1 else ''  # Current Price
        row[2] = parts[2] if len(parts) > 2 else ''  # Market Cap 
        row[3] = parts[3] if len(parts) > 3 else ''  # Volume or other data
        row[6] = parts[4] if len(parts) > 4 else ''  # Thesis
        row[18] = source  # AI Source
        row[19] = timestamp  # Timestamp
        
        return row
    
    def write_to_sheet(self, openai_content: Optional[str], claude_content: Optional[str]) -> bool:
        """Write both AI research results to Google Sheets"""
        try:
            all_rows = []
            
            # First, ensure we have proper headers
            headers = [
                "Ticker", "Current Price", "Market Cap", "Avg Daily Volume", "Setup Type", "Catalyst & Date", 
                "Thesis", "Entry Zone", "Stop Loss", "Target T1", "Target T2", "Target T3",
                "Position Size %", "Risk:Reward Ratio", "Expected 5-Day Move %", 
                "Confidence (1-5)", "Key Risks", "Options Alternative", "Execution Notes",
                "AI Source", "Generated Timestamp"
            ]
            
            # Check if headers exist, if not create them
            try:
                result = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{self.sheet_name}!A1:U1"
                ).execute()
                
                existing_headers = result.get('values', [[]])
                if not existing_headers or not existing_headers[0]:
                    # Write headers
                    self.sheets_service.spreadsheets().values().update(
                        spreadsheetId=self.spreadsheet_id,
                        range=f"{self.sheet_name}!A1:U1",
                        valueInputOption='RAW',
                        body={'values': [headers]}
                    ).execute()
                    logger.info("Headers created successfully")
            except Exception as e:
                logger.warning(f"Could not check/create headers: {e}")
            
            # Add OpenAI content if available
            if openai_content:
                openai_rows = self.parse_research_to_rows(openai_content, "OpenAI")
                all_rows.extend(openai_rows)
            
            # Add Claude content if available
            if claude_content:
                claude_rows = self.parse_research_to_rows(claude_content, "Claude")
                all_rows.extend(claude_rows)
            
            if not all_rows:
                logger.error("No content to write to sheet")
                return False
            
            # Write to sheet starting from row 2 (after header)
            if len(all_rows[0]) == 21:  # Structured data (updated for 21 columns)
                range_name = f"{self.sheet_name}!A2:U{len(all_rows) + 1}"
            else:  # Free-form text
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
            
            logger.info(f"Successfully wrote {len(all_rows)} rows to sheet")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to sheet: {e}")
            return False
    
    def process_weekly_research(self) -> Dict[str, Any]:
        """Main processing function for weekly research"""
        results = {
            'success': False,
            'openai_success': False,
            'claude_success': False,
            'sheet_cleared': False,
            'sheet_updated': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        try:
            # Step 1: Clear the sheet except header
            results['sheet_cleared'] = self.clear_sheet_except_header()
            
            # Step 2: Generate research from both AIs in parallel
            openai_content = None
            claude_content = None
            
            try:
                openai_content = self.get_openai_research()
                results['openai_success'] = openai_content is not None
            except Exception as e:
                results['errors'].append(f"OpenAI error: {str(e)}")
            
            try:
                claude_content = self.get_claude_research()
                results['claude_success'] = claude_content is not None
            except Exception as e:
                results['errors'].append(f"Claude error: {str(e)}")
            
            # Step 3: Write results to sheet
            if openai_content or claude_content:
                results['sheet_updated'] = self.write_to_sheet(openai_content, claude_content)
            
            # Overall success if at least one AI succeeded and sheet was updated
            results['success'] = (results['openai_success'] or results['claude_success']) and results['sheet_updated']
            
        except Exception as e:
            logger.error(f"Critical error in weekly research processing: {e}")
            results['errors'].append(f"Critical error: {str(e)}")
        
        return results


def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Weekly research Lambda function started")
    
    try:
        # Initialize processor
        processor = WeeklyResearchProcessor()
        
        # Process weekly research
        results = processor.process_weekly_research()
        
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
    
    # Mock event and context for testing
    test_event = {}
    test_context = type('Context', (), {
        'function_name': 'weekly_research_lambda',
        'function_version': '1',
        'invoked_function_arn': 'arn:aws:lambda:us-east-1:123456789:function:weekly_research_lambda',
        'memory_limit_in_mb': '512',
        'remaining_time_in_millis': lambda: 30000
    })()
    
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2))