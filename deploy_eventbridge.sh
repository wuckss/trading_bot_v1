#!/bin/bash

# EventBridge Schedule Deployment Script for AI Trading Bot
# This script creates EventBridge schedules for weekly research and daily validation

set -e

# Configuration
WEEKLY_LAMBDA_NAME="weekly_research_lambda"
DAILY_LAMBDA_NAME="daily_validation_lambda"
ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/EventBridgeExecutionRole"
REGION="us-east-1"

echo "🚀 Deploying EventBridge schedules for AI Trading Bot..."

# Function to create schedule
create_schedule() {
    local schedule_name=$1
    local description=$2
    local cron_expression=$3
    local lambda_name=$4
    local input_json=$5
    
    echo "Creating schedule: $schedule_name"
    
    aws scheduler create-schedule \
        --name "$schedule_name" \
        --description "$description" \
        --schedule-expression "$cron_expression" \
        --schedule-expression-timezone "America/New_York" \
        --target "{
            \"Arn\": \"arn:aws:lambda:$REGION:$(aws sts get-caller-identity --query Account --output text):function:$lambda_name\",
            \"RoleArn\": \"$ROLE_ARN\",
            \"Input\": \"$input_json\"
        }" \
        --flexible-time-window "{
            \"Mode\": \"FLEXIBLE\",
            \"MaximumWindowInMinutes\": 30
        }" \
        --retry-policy "{
            \"MaximumRetryAttempts\": 2
        }" \
        --region "$REGION"
    
    echo "✅ Created: $schedule_name"
}

# 1. Weekly Research Schedule - Sunday 8:00 AM ET
create_schedule \
    "weekly-research-schedule" \
    "Weekly AI trading research - Sundays 8:00 AM ET" \
    "cron(0 13 ? * SUN *)" \
    "$WEEKLY_LAMBDA_NAME" \
    '{"source":"eventbridge","schedule_type":"weekly"}'

# 2. Daily AM Validation Schedule - Mon-Fri 8:00 AM ET  
create_schedule \
    "daily-validation-am-schedule" \
    "Daily AM validation - Mon-Fri 8:00 AM ET (premarket)" \
    "cron(0 13 ? * MON-FRI *)" \
    "$DAILY_LAMBDA_NAME" \
    '{"session_type":"AM","source":"eventbridge","schedule_type":"daily_am"}'

# 3. Daily PM Validation Schedule - Mon-Fri 3:30 PM ET
create_schedule \
    "daily-validation-pm-schedule" \
    "Daily PM validation - Mon-Fri 3:30 PM ET (post-market)" \
    "cron(30 20 ? * MON-FRI *)" \
    "$DAILY_LAMBDA_NAME" \
    '{"session_type":"PM","source":"eventbridge","schedule_type":"daily_pm"}'

echo ""
echo "🎉 All EventBridge schedules created successfully!"
echo ""
echo "📊 Schedule Summary:"
echo "  - Weekly Research: Sundays 8:00 AM ET"
echo "  - Daily AM Validation: Mon-Fri 8:00 AM ET"  
echo "  - Daily PM Validation: Mon-Fri 3:30 PM ET"
echo ""
echo "⚙️  Next Steps:"
echo "  1. Update ROLE_ARN in this script with your actual IAM role"
echo "  2. Ensure Lambda functions are deployed with the correct names"
echo "  3. Test schedules with: aws scheduler get-schedule --name <schedule-name>"
echo "  4. Monitor CloudWatch logs for execution results"
echo ""
echo "🔒 Security Notes:"  
echo "  - EventBridge uses IAM role for Lambda execution"
echo "  - Functions include built-in market calendar and idempotency checks"
echo "  - All schedules respect NYSE trading calendar"