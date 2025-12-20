"""Allow list storage module for managing MCP tool permissions."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class AllowListStorage:
    """Allow list storage using SQLite database.
    
    Stores allowed MCP tool names in a SQLite database.
    Database file location: {thread_storage_dir}/.allow_list.db
    """

    def __init__(self, thread_storage_dir: Path):
        """Initialize allow list storage.
        
        Args:
            thread_storage_dir: Directory where the database file will be stored
        """
        self.storage_dir = Path(thread_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / ".allow_list.db"
        self.initialize_db()

    def initialize_db(self) -> None:
        """Initialize database table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create allow_list table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS allow_list (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL UNIQUE,
                    created_at INTEGER NOT NULL
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_allow_list_tool_name 
                ON allow_list(tool_name)
            """)
            
            conn.commit()

    def add_tool(self, tool_name: str) -> bool:
        """Add a tool to the allow list.
        
        Args:
            tool_name: Name of the tool to add
            
        Returns:
            True if added successfully, False if already exists
        """
        current_time = int(time.time())
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO allow_list (tool_name, created_at)
                    VALUES (?, ?)
                """, (tool_name, current_time))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Tool already exists
            return False

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the allow list.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM allow_list WHERE tool_name = ?", (tool_name,))
            conn.commit()
            return cursor.rowcount > 0

    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is in the allow list.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is in allow list, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM allow_list WHERE tool_name = ?", (tool_name,))
            return cursor.fetchone() is not None

    def list_allowed_tools(self) -> list[str]:
        """List all tools in the allow list.
        
        Returns:
            List of tool names in the allow list
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tool_name FROM allow_list ORDER BY created_at")
            return [row[0] for row in cursor.fetchall()]
