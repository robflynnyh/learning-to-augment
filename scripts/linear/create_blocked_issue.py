#!/usr/bin/env python3
"""Create a Linear follow-up issue blocked by the current issue.

Use this for concrete problems discovered during Symphony work that are real
but outside the current issue scope. The new issue is created in the same team
and project as the blocking issue, then linked with an IssueRelationType.blocks
relation from the current issue to the new issue.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


LINEAR_API_URL = "https://api.linear.app/graphql"


class LinearError(RuntimeError):
    pass


def graphql(api_key: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        LINEAR_API_URL,
        data=payload,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "User-Agent": "learning-to-augment-issue-helper",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise LinearError(f"Linear API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise LinearError(f"Linear API request failed: {error}") from error

    decoded = json.loads(body)
    if decoded.get("errors"):
        raise LinearError(json.dumps(decoded["errors"], indent=2))
    return decoded["data"]


def fetch_issue(api_key: str, issue: str) -> Dict[str, Any]:
    query = """
    query BlockingIssue($id: String!) {
      issue(id: $id) {
        id
        identifier
        title
        url
        team {
          id
          states(first: 100) {
            nodes {
              id
              name
            }
          }
        }
        project {
          id
          name
        }
      }
    }
    """
    data = graphql(api_key, query, {"id": issue})
    found = data.get("issue")
    if not found:
        raise LinearError(f"Linear issue not found: {issue}")
    return found


def state_id(issue: Dict[str, Any], target_state: Optional[str]) -> Optional[str]:
    if not target_state:
        return None
    states = issue["team"]["states"]["nodes"]
    for state in states:
        if state["name"].lower() == target_state.lower():
            return state["id"]
    available = ", ".join(sorted(state["name"] for state in states))
    raise LinearError(f"State {target_state!r} not found. Available states: {available}")


def read_description(args: argparse.Namespace) -> str:
    pieces = []
    if args.description:
        pieces.append(args.description)
    if args.description_file:
        try:
            pieces.append(Path(args.description_file).read_text())
        except OSError as error:
            raise LinearError(f"Could not read description file: {error}") from error
    return "\n\n".join(piece.strip() for piece in pieces if piece.strip())


def build_description(args: argparse.Namespace, blocker: Dict[str, Any]) -> str:
    description = read_description(args)
    prefix = (
        f"Created by Symphony while working on {blocker['identifier']}.\n\n"
        f"Blocked by: {blocker['url']}\n\n"
        "Scope note: this looked like a separate actionable problem and should "
        "not be folded into the blocking issue unless a human says otherwise."
    )
    if description:
        return f"{prefix}\n\n---\n\n{description}"
    return prefix


def create_issue(api_key: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    mutation = """
    mutation CreateBlockedIssue($input: IssueCreateInput!) {
      issueCreate(input: $input) {
        success
        issue {
          id
          identifier
          title
          url
        }
      }
    }
    """
    data = graphql(api_key, mutation, {"input": input_data})
    result = data["issueCreate"]
    if not result["success"]:
        raise LinearError("Linear issueCreate returned success=false")
    return result["issue"]


def create_blocks_relation(api_key: str, blocker_id: str, blocked_id: str) -> None:
    mutation = """
    mutation CreateBlockedRelation($input: IssueRelationCreateInput!) {
      issueRelationCreate(input: $input) {
        success
      }
    }
    """
    data = graphql(
        api_key,
        mutation,
        {
            "input": {
                "issueId": blocker_id,
                "relatedIssueId": blocked_id,
                "type": "blocks",
            }
        },
    )
    if not data["issueRelationCreate"]["success"]:
        raise LinearError("Linear issueRelationCreate returned success=false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocked-by",
        required=True,
        help="Current Linear issue id or identifier, for example ROB-123.",
    )
    parser.add_argument("--title", required=True, help="Title for the new issue.")
    parser.add_argument("--description", help="Markdown description for the new issue.")
    parser.add_argument("--description-file", help="Read extra Markdown description from a file.")
    parser.add_argument(
        "--state",
        default="Todo",
        help="Initial workflow state for the new issue. Use an empty string for Linear default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the issue payload and relation that would be created.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        raise LinearError("LINEAR_API_KEY is required")

    blocker = fetch_issue(api_key, args.blocked_by)
    target_state_id = state_id(blocker, args.state)
    input_data = {
        "teamId": blocker["team"]["id"],
        "title": args.title,
        "description": build_description(args, blocker),
    }
    if blocker.get("project"):
        input_data["projectId"] = blocker["project"]["id"]
    if target_state_id:
        input_data["stateId"] = target_state_id

    if args.dry_run:
        print(
            json.dumps(
                {
                    "issueCreate": input_data,
                    "relation": {
                        "issueId": blocker["id"],
                        "relatedIssueId": "<new issue id>",
                        "type": "blocks",
                    },
                    "blockedBy": {
                        "identifier": blocker["identifier"],
                        "url": blocker["url"],
                    },
                },
                indent=2,
            )
        )
        return 0

    created = create_issue(api_key, input_data)
    create_blocks_relation(api_key, blocker["id"], created["id"])
    print(f"Created {created['identifier']}: {created['url']}")
    print(f"Blocked by {blocker['identifier']}: {blocker['url']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LinearError as error:
        print(f"create_blocked_issue: {error}", file=sys.stderr)
        raise SystemExit(1)
