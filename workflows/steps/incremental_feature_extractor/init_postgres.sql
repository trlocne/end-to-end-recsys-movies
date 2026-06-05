-- Create table for tracking incremental processing state
CREATE TABLE IF NOT EXISTS incremental_state (
    component VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_incremental_state_component ON incremental_state(component);

-- Insert initial record for feature extractor if not exists
INSERT INTO incremental_state (component, timestamp, updated_at)
VALUES ('feature_extractor', '1970-01-01T00:00:00', CURRENT_TIMESTAMP)
ON CONFLICT (component) DO NOTHING;

-- Interactions table (CDC source)
CREATE TABLE IF NOT EXISTS interactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    rating FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    interaction_type VARCHAR(50) DEFAULT 'rating'
);

-- Users table (CDC source)
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Items table (CDC source)
CREATE TABLE IF NOT EXISTS items (
    item_id BIGINT PRIMARY KEY,
    title VARCHAR(255),
    genre JSONB,
    tags JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_interactions_user_item ON interactions(user_id, item_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
