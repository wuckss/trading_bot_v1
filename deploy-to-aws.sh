#!/bin/bash

# =============================================================================
# AI Trading Bot - Complete AWS Deployment Script
# =============================================================================
# This script deploys the AI Trading Bot Lambda functions with EventBridge 
# scheduling, IAM roles, and CloudWatch logging to AWS.
#
# Prerequisites:
# - AWS CLI configured with appropriate permissions
# - API credentials ready (OpenAI, Anthropic, Google, Telegram)
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-trading-bot"
REGION="us-east-1"  # Change if needed
PYTHON_VERSION="python3.9"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}🚀 AI Trading Bot - AWS Deployment Script${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# =============================================================================
# Step 1: Environment Variables Setup
# =============================================================================
echo -e "\n${YELLOW}📝 Step 1: Environment Variables Setup${NC}"

# Function to prompt for environment variables
prompt_env_var() {
    local var_name="$1"
    local description="$2"
    local is_secret="$3"
    
    if [[ -n "${!var_name}" ]]; then
        echo -e "${GREEN}✓ $var_name already set${NC}"
        return
    fi
    
    echo -e "${BLUE}Enter $description:${NC}"
    if [[ "$is_secret" == "true" ]]; then
        read -s value
        echo
    else
        read value
    fi
    
    export "$var_name"="$value"
    echo -e "${GREEN}✓ $var_name configured${NC}"
}

echo "Setting up required environment variables..."

# API Keys
prompt_env_var "OPENAI_API_KEY" "OpenAI API Key (sk-...)" "true"
prompt_env_var "ANTHROPIC_API_KEY" "Anthropic API Key (sk-ant-...)" "true"

# Google Sheets Configuration  
echo -e "\n${BLUE}For Google Sheets integration, you'll need a service account JSON.${NC}"
echo -e "${BLUE}Get it from: https://console.cloud.google.com/iam-admin/serviceaccounts${NC}"
prompt_env_var "GOOGLE_SERVICE_ACCOUNT_JSON" "Google Service Account JSON (entire JSON object)" "false"
prompt_env_var "GOOGLE_SHEETS_WEEKLY_ID" "Google Sheets Weekly Research ID" "false"
prompt_env_var "GOOGLE_SHEETS_DAILY_ID" "Google Sheets Daily Validation ID" "false"

# Sheet Names (with defaults)
export WEEKLY_SHEET_NAME="Weekly_Research"
export DAILY_AM_SHEET_NAME="Daily_AM"  
export DAILY_PM_SHEET_NAME="Daily_PM"

# Model Configuration (with defaults)
export OPENAI_MODEL="gpt-5"
export CLAUDE_MODEL="claude-opus-4-1"
export ENABLE_DEEP_RESEARCH="true"

# Telegram Configuration
echo -e "\n${BLUE}For Telegram notifications (optional - press Enter to skip):${NC}"
echo -e "${BLUE}Create bot: https://t.me/BotFather | Get chat ID: https://t.me/userinfobot${NC}"
read -p "Telegram Bot Token (optional): " TELEGRAM_BOT_TOKEN
read -p "Telegram Chat ID (optional): " TELEGRAM_CHAT_ID

export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

echo -e "${GREEN}✅ Environment variables configured${NC}"

# =============================================================================
# Step 2: Create IAM Role for Lambda
# =============================================================================
echo -e "\n${YELLOW}🔐 Step 2: Creating IAM Role${NC}"

ROLE_NAME="${PROJECT_NAME}-lambda-role"

# Check if role exists
if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo -e "${GREEN}✓ IAM role '$ROLE_NAME' already exists${NC}"
else
    echo "Creating IAM role..."
    
    # Trust policy for Lambda
    cat > trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    # Create the role
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file://trust-policy.json
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    
    rm trust-policy.json
    echo -e "${GREEN}✅ IAM role created successfully${NC}"
fi

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
echo -e "${BLUE}📋 Role ARN: $ROLE_ARN${NC}"

# =============================================================================
# Step 3: Prepare Lambda Deployment Packages
# =============================================================================
echo -e "\n${YELLOW}📦 Step 3: Creating Lambda Deployment Packages${NC}"

# Create temporary directory for builds
BUILD_DIR="lambda-builds"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Function to create deployment package
create_lambda_package() {
    local function_name="$1"
    local python_file="$2"
    local package_dir="$BUILD_DIR/${function_name}"
    
    echo "Creating package for $function_name..."
    
    # Create package directory
    mkdir -p "$package_dir"
    
    # Install dependencies
    pip install -r requirements.txt -t "$package_dir" --quiet
    
    # Copy Lambda function
    cp "$python_file" "$package_dir/lambda_function.py"
    
    # Create ZIP package
    cd "$package_dir"
    zip -r "../${function_name}.zip" . --quiet
    cd - &>/dev/null
    
    echo -e "${GREEN}✓ Package created: ${function_name}.zip${NC}"
}

# Create packages for both functions
create_lambda_package "weekly-research" "weekly_research_lambda.py"
create_lambda_package "daily-validation" "daily_validation_lambda.py"

# =============================================================================
# Step 4: Deploy Lambda Functions  
# =============================================================================
echo -e "\n${YELLOW}🚀 Step 4: Deploying Lambda Functions${NC}"

# Function to deploy Lambda
deploy_lambda() {
    local function_name="$1"
    local description="$2"
    local timeout="$3"
    local memory="$4"
    
    local full_name="${PROJECT_NAME}-${function_name}"
    local zip_file="$BUILD_DIR/${function_name}.zip"
    
    echo "Deploying $full_name..."
    
    # Check if function exists
    if aws lambda get-function --function-name "$full_name" &>/dev/null; then
        echo "Updating existing function..."
        aws lambda update-function-code \
            --function-name "$full_name" \
            --zip-file "fileb://$zip_file" &>/dev/null
    else
        echo "Creating new function..."
        aws lambda create-function \
            --function-name "$full_name" \
            --runtime "$PYTHON_VERSION" \
            --role "$ROLE_ARN" \
            --handler "lambda_function.lambda_handler" \
            --zip-file "fileb://$zip_file" \
            --description "$description" \
            --timeout "$timeout" \
            --memory-size "$memory" &>/dev/null
    fi
    
    # Update environment variables
    aws lambda update-function-configuration \
        --function-name "$full_name" \
        --environment Variables="{\
OPENAI_API_KEY=\"$OPENAI_API_KEY\",\
ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\",\
GOOGLE_SERVICE_ACCOUNT_JSON=\"$GOOGLE_SERVICE_ACCOUNT_JSON\",\
GOOGLE_SHEETS_WEEKLY_ID=\"$GOOGLE_SHEETS_WEEKLY_ID\",\
GOOGLE_SHEETS_DAILY_ID=\"$GOOGLE_SHEETS_DAILY_ID\",\
WEEKLY_SHEET_NAME=\"$WEEKLY_SHEET_NAME\",\
DAILY_AM_SHEET_NAME=\"$DAILY_AM_SHEET_NAME\",\
DAILY_PM_SHEET_NAME=\"$DAILY_PM_SHEET_NAME\",\
OPENAI_MODEL=\"$OPENAI_MODEL\",\
CLAUDE_MODEL=\"$CLAUDE_MODEL\",\
ENABLE_DEEP_RESEARCH=\"$ENABLE_DEEP_RESEARCH\",\
TELEGRAM_BOT_TOKEN=\"$TELEGRAM_BOT_TOKEN\",\
TELEGRAM_CHAT_ID=\"$TELEGRAM_CHAT_ID\"\
}" &>/dev/null
    
    echo -e "${GREEN}✅ $full_name deployed successfully${NC}"
}

# Deploy functions
deploy_lambda "weekly-research" "Weekly AI Trading Research Generator" 600 1024
deploy_lambda "daily-validation" "Daily Trade Validation and Narrowing" 300 512

# =============================================================================
# Step 5: Create EventBridge Rules
# =============================================================================
echo -e "\n${YELLOW}⏰ Step 5: Setting up EventBridge Scheduling${NC}"

# Function to create EventBridge rule
create_eventbridge_rule() {
    local rule_name="$1"
    local function_name="$2"
    local schedule_expression="$3"
    local description="$4"
    local event_input="$5"
    
    local full_function_name="${PROJECT_NAME}-${function_name}"
    local full_rule_name="${PROJECT_NAME}-${rule_name}"
    
    echo "Creating EventBridge rule: $full_rule_name..."
    
    # Create or update the rule
    aws events put-rule \
        --name "$full_rule_name" \
        --schedule-expression "$schedule_expression" \
        --description "$description" \
        --state ENABLED &>/dev/null
    
    # Get function ARN
    FUNCTION_ARN=$(aws lambda get-function --function-name "$full_function_name" --query 'Configuration.FunctionArn' --output text)
    
    # Add Lambda target to rule
    aws events put-targets \
        --rule "$full_rule_name" \
        --targets "Id=1,Arn=$FUNCTION_ARN,Input='$event_input'" &>/dev/null
    
    # Add permission for EventBridge to invoke Lambda
    aws lambda add-permission \
        --function-name "$full_function_name" \
        --statement-id "eventbridge-$rule_name" \
        --action lambda:InvokeFunction \
        --principal events.amazonaws.com \
        --source-arn "arn:aws:events:$REGION:$(aws sts get-caller-identity --query Account --output text):rule/$full_rule_name" &>/dev/null || true
    
    echo -e "${GREEN}✓ EventBridge rule created: $full_rule_name${NC}"
}

# Create scheduling rules
echo "Setting up automated scheduling..."

# Weekly Research: Sundays at 8:00 PM UTC (after US market close)
create_eventbridge_rule "weekly-schedule" "weekly-research" \
    "cron(0 20 ? * SUN *)" \
    "Weekly AI Trading Research - Runs Sundays at 8 PM UTC" \
    '{}'

# Daily AM Validation: Monday-Friday at 12:00 UTC (8:00 AM ET)
create_eventbridge_rule "daily-am-schedule" "daily-validation" \
    "cron(0 12 * * MON-FRI *)" \
    "Daily AM Trade Validation - Runs weekdays at 8 AM ET" \
    '{"session_type":"AM"}'

# Daily PM Validation: Monday-Friday at 20:00 UTC (3:00 PM ET) 
create_eventbridge_rule "daily-pm-schedule" "daily-validation" \
    "cron(0 20 * * MON-FRI *)" \
    "Daily PM Trade Validation - Runs weekdays at 3 PM ET" \
    '{"session_type":"PM"}'

echo -e "${GREEN}✅ All EventBridge rules created${NC}"

# =============================================================================
# Step 6: Setup CloudWatch Log Groups
# =============================================================================
echo -e "\n${YELLOW}📊 Step 6: Setting up CloudWatch Logging${NC}"

# Function to create log group
create_log_group() {
    local function_name="$1"
    local retention_days="$2"
    
    local full_function_name="${PROJECT_NAME}-${function_name}"
    local log_group_name="/aws/lambda/$full_function_name"
    
    # Check if log group exists
    if aws logs describe-log-groups --log-group-name-prefix "$log_group_name" --query 'logGroups[0].logGroupName' --output text | grep -q "$log_group_name"; then
        echo -e "${GREEN}✓ Log group already exists: $log_group_name${NC}"
    else
        echo "Creating log group: $log_group_name..."
        aws logs create-log-group --log-group-name "$log_group_name" &>/dev/null
        echo -e "${GREEN}✓ Log group created: $log_group_name${NC}"
    fi
    
    # Set retention policy
    aws logs put-retention-policy \
        --log-group-name "$log_group_name" \
        --retention-in-days "$retention_days" &>/dev/null
    
    echo -e "${BLUE}📋 Log retention set to $retention_days days${NC}"
}

# Create log groups with 30-day retention
create_log_group "weekly-research" 30
create_log_group "daily-validation" 30

# =============================================================================
# Step 7: Test Deployment
# =============================================================================
echo -e "\n${YELLOW}🧪 Step 7: Testing Deployment${NC}"

echo "Testing Lambda function invocations..."

# Test weekly research function (dry run)
echo "Testing weekly research function..."
aws lambda invoke \
    --function-name "${PROJECT_NAME}-weekly-research" \
    --payload '{"test": true}' \
    --output json \
    response-weekly.json &>/dev/null

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Weekly research function test passed${NC}"
else
    echo -e "${RED}✗ Weekly research function test failed${NC}"
fi

# Test daily validation function (dry run)
echo "Testing daily validation function..."
aws lambda invoke \
    --function-name "${PROJECT_NAME}-daily-validation" \
    --payload '{"session_type": "AM", "test": true}' \
    --output json \
    response-daily.json &>/dev/null

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Daily validation function test passed${NC}"
else
    echo -e "${RED}✗ Daily validation function test failed${NC}"
fi

# Clean up test files
rm -f response-weekly.json response-daily.json

# =============================================================================
# Step 8: Cleanup and Summary
# =============================================================================
echo -e "\n${YELLOW}🧹 Step 8: Cleanup${NC}"

# Remove build artifacts
rm -rf "$BUILD_DIR"
echo -e "${GREEN}✓ Build artifacts cleaned up${NC}"

# =============================================================================
# Deployment Summary
# =============================================================================
echo -e "\n${GREEN}==============================================================================${NC}"
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉${NC}"
echo -e "${GREEN}==============================================================================${NC}"

echo -e "\n${BLUE}📋 Deployed Resources:${NC}"
echo -e "${BLUE}├── Lambda Functions:${NC}"
echo -e "${BLUE}│   ├── ${PROJECT_NAME}-weekly-research${NC}"
echo -e "${BLUE}│   └── ${PROJECT_NAME}-daily-validation${NC}"
echo -e "${BLUE}├── EventBridge Rules:${NC}" 
echo -e "${BLUE}│   ├── ${PROJECT_NAME}-weekly-schedule (Sundays 8PM UTC)${NC}"
echo -e "${BLUE}│   ├── ${PROJECT_NAME}-daily-am-schedule (Weekdays 8AM ET)${NC}"
echo -e "${BLUE}│   └── ${PROJECT_NAME}-daily-pm-schedule (Weekdays 3PM ET)${NC}"
echo -e "${BLUE}├── IAM Role: ${ROLE_NAME}${NC}"
echo -e "${BLUE}└── CloudWatch Log Groups (30-day retention)${NC}"

echo -e "\n${BLUE}⏰ Schedule Summary:${NC}"
echo -e "${BLUE}├── Weekly Research: Sundays at 8:00 PM UTC${NC}"
echo -e "${BLUE}├── Daily AM Validation: Weekdays at 8:00 AM ET (12:00 UTC)${NC}"
echo -e "${BLUE}└── Daily PM Validation: Weekdays at 3:00 PM ET (20:00 UTC)${NC}"

echo -e "\n${BLUE}🔗 Next Steps:${NC}"
echo -e "${BLUE}1. Check CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region=$REGION#logsV2:log-groups${NC}"
echo -e "${BLUE}2. Monitor Lambda Functions: https://console.aws.amazon.com/lambda/home?region=$REGION#/functions${NC}"
echo -e "${BLUE}3. View EventBridge Rules: https://console.aws.amazon.com/events/home?region=$REGION#/rules${NC}"
echo -e "${BLUE}4. Test manual execution: aws lambda invoke --function-name ${PROJECT_NAME}-weekly-research test-output.json${NC}"

if [[ -n "$TELEGRAM_BOT_TOKEN" ]]; then
    echo -e "${BLUE}5. Telegram notifications configured ✅${NC}"
else
    echo -e "${YELLOW}5. Telegram notifications not configured (optional)${NC}"
fi

echo -e "\n${GREEN}🎯 Your AI Trading Bot is now fully automated on AWS! 🎯${NC}"
echo -e "${GREEN}The system will automatically generate weekly research and daily validations.${NC}"
echo -e "${GREEN}==============================================================================${NC}"