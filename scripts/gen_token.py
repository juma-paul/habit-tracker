"""Generate a signed JWT for manual API testing.

Usage:
    uv run python scripts/gen_token.py
    uv run python scripts/gen_token.py --email you@example.com
"""
import argparse
import os
import sys

import jwt
from dotenv import load_dotenv

load_dotenv()

secret = os.getenv("JWT_SECRET")
if not secret:
    print("ERROR: JWT_SECRET not set in .env", file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--user-id", default="00000000-0000-0000-0000-000000000001", help="AuthKit userId claim")
parser.add_argument("--email", default="test@example.com", help="email claim")
args = parser.parse_args()

token = jwt.encode(
    {"userId": args.user_id, "email": args.email},
    secret,
    algorithm="HS256",
)

print(token)
