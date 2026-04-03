-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Habits table
CREATE TABLE IF NOT EXISTS habits (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    target FLOAT,
    unit VARCHAR(50),
    frequency VARCHAR(20) DEFAULT 'daily',
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Habit logs table
CREATE TABLE IF NOT EXISTS habit_logs (
    id SERIAL PRIMARY KEY,
    habit_id INTEGER REFERENCES habits(id) ON DELETE CASCADE,
    value FLOAT NOT NULL,
    notes TEXT,
    logged_at TIMESTAMP DEFAULT NOW()
);

-- Conversations table (chat sessions)
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
); 

-- Messages table (chat history)
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User settings table
CREATE TABLE IF NOT EXISTS user_settings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(20) DEFAULT 'system',
    voice_enabled BOOLEAN DEFAULT TRUE,
    notifications BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_habits_user ON habits(user_id);
CREATE INDEX IF NOT EXISTS idx_habits_user_active ON habits(user_id) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_logs_habit ON habit_logs(habit_id);
CREATE INDEX IF NOT EXISTS idx_logs_habit_date ON habit_logs(habit_id, logged_at);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);

-- DEVELOPMENT: seed test user(id=1)
INSERT INTO users (email, name)
VALUES ('dev@example.com', 'Dev User')
ON CONFLICT (email) DO NOTHING;