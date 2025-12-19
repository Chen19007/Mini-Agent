"""Thread storage module for saving and loading ACP session history."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from mini_agent.schema import Message, ToolCall


class ThreadStorage:
    """Thread storage using SQLite database.
    
    Stores thread metadata and messages in a SQLite database.
    Database file location: {thread_storage_dir}/.threads.db
    """

    def __init__(self, thread_storage_dir: Path):
        """Initialize thread storage.
        
        Args:
            thread_storage_dir: Directory where the database file will be stored
        """
        self.storage_dir = Path(thread_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / ".threads.db"
        self.initialize_db()

    def initialize_db(self) -> None:
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create threads table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    workspace_dir TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    thinking TEXT,
                    tool_calls_json TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_thread_id 
                ON messages(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_sequence 
                ON messages(thread_id, sequence)
            """)
            
            conn.commit()

    def save_thread(
        self,
        thread_id: str,
        messages: list[Message],
        workspace_dir: str,
        title: str | None = None,
    ) -> None:
        """Save or update a thread with its messages.
        
        Args:
            thread_id: Thread/session ID
            messages: List of messages to save
            workspace_dir: Workspace directory path
            title: Optional thread title (if None, will try to extract from first user message)
        """
        if not messages:
            return
        
        # Extract title from first user message if not provided
        if title is None:
            for msg in messages:
                if msg.role == "user":
                    content = msg.content
                    if isinstance(content, str):
                        # Use first 100 characters as title
                        title = content[:100].strip()
                        if len(content) > 100:
                            title += "..."
                    break
        
        if title is None:
            title = "Untitled Thread"
        
        current_time = int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if thread exists
            cursor.execute("SELECT id FROM threads WHERE id = ?", (thread_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing thread
                cursor.execute("""
                    UPDATE threads 
                    SET title = ?, workspace_dir = ?, updated_at = ?
                    WHERE id = ?
                """, (title, workspace_dir, current_time, thread_id))
                
                # Delete existing messages
                cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            else:
                # Insert new thread
                cursor.execute("""
                    INSERT INTO threads (id, title, workspace_dir, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (thread_id, title, workspace_dir, current_time, current_time))
            
            # Insert messages
            for sequence, msg in enumerate(messages):
                serialized = self._serialize_message(msg)
                cursor.execute("""
                    INSERT INTO messages (
                        thread_id, sequence, role, content, thinking,
                        tool_calls_json, tool_call_id, name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    thread_id,
                    sequence,
                    serialized["role"],
                    serialized["content"],
                    serialized.get("thinking"),
                    serialized.get("tool_calls_json"),
                    serialized.get("tool_call_id"),
                    serialized.get("name"),
                ))
            
            conn.commit()

    def load_thread(self, thread_id: str) -> tuple[list[Message], str] | None:
        """Load a thread and its messages.
        
        Args:
            thread_id: Thread/session ID
            
        Returns:
            Tuple of (messages list, workspace_dir) if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get thread metadata
            cursor.execute("SELECT workspace_dir FROM threads WHERE id = ?", (thread_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            
            workspace_dir = row["workspace_dir"]
            
            # Get messages
            cursor.execute("""
                SELECT role, content, thinking, tool_calls_json, tool_call_id, name
                FROM messages
                WHERE thread_id = ?
                ORDER BY sequence
            """, (thread_id,))
            
            messages = []
            for row in cursor.fetchall():
                msg_data = {
                    "role": row["role"],
                    "content": row["content"],
                    "thinking": row["thinking"],
                    "tool_calls_json": row["tool_calls_json"],
                    "tool_call_id": row["tool_call_id"],
                    "name": row["name"],
                }
                msg = self._deserialize_message(msg_data)
                messages.append(msg)
            
            return messages, workspace_dir

    def list_threads(self, workspace_dir: str | None = None) -> list[dict[str, Any]]:
        """List all threads, optionally filtered by workspace.
        
        Args:
            workspace_dir: Optional workspace directory to filter by
            
        Returns:
            List of thread metadata dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if workspace_dir:
                # Normalize workspace_dir for comparison
                workspace_dir_normalized = str(Path(workspace_dir).resolve())
                
                # Get all threads and filter by normalized path
                cursor.execute("""
                    SELECT id, title, workspace_dir, created_at, updated_at
                    FROM threads
                    ORDER BY updated_at DESC
                """)
                
                threads = []
                for row in cursor.fetchall():
                    # Normalize stored workspace_dir for comparison
                    stored_workspace = str(Path(row["workspace_dir"]).resolve())
                    if stored_workspace == workspace_dir_normalized:
                        threads.append({
                            "id": row["id"],
                            "title": row["title"],
                            "workspace_dir": row["workspace_dir"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        })
            else:
                cursor.execute("""
                    SELECT id, title, workspace_dir, created_at, updated_at
                    FROM threads
                    ORDER BY updated_at DESC
                """)
                
                threads = []
                for row in cursor.fetchall():
                    threads.append({
                        "id": row["id"],
                        "title": row["title"],
                        "workspace_dir": row["workspace_dir"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    })
            
            return threads

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread and all its messages.
        
        Args:
            thread_id: Thread/session ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            conn.commit()

    def _serialize_message(self, msg: Message) -> dict[str, Any]:
        """Serialize a Message object to a dictionary for database storage.
        
        Args:
            msg: Message object to serialize
            
        Returns:
            Dictionary with serialized message data
        """
        # Serialize content: if it's a list or dict, convert to JSON string; otherwise keep as string
        if isinstance(msg.content, (list, dict)):
            content_str = json.dumps(msg.content, ensure_ascii=False)
        else:
            content_str = str(msg.content) if msg.content is not None else ""
        
        result = {
            "role": msg.role,
            "content": content_str,
            "thinking": msg.thinking,
            "tool_call_id": msg.tool_call_id,
            "name": msg.name,
        }
        
        # Serialize tool_calls
        if msg.tool_calls:
            tool_calls_data = []
            for tc in msg.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            result["tool_calls_json"] = json.dumps(tool_calls_data, ensure_ascii=False)
        else:
            result["tool_calls_json"] = None
        
        return result

    def _deserialize_message(self, data: dict[str, Any]) -> Message:
        """Deserialize a dictionary to a Message object.
        
        Args:
            data: Dictionary with message data from database
            
        Returns:
            Message object
        """
        # Deserialize content
        content = data["content"]
        try:
            # Try to parse as JSON (if it was a list/dict)
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, use as string
            pass
        
        # Deserialize tool_calls
        tool_calls = None
        if data.get("tool_calls_json"):
            try:
                tool_calls_data = json.loads(data["tool_calls_json"])
                tool_calls = []
                for tc_data in tool_calls_data:
                    tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        type=tc_data["type"],
                        function=tc_data["function"],
                    ))
            except (json.JSONDecodeError, KeyError, TypeError):
                tool_calls = None
        
        return Message(
            role=data["role"],
            content=content,
            thinking=data.get("thinking"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )
