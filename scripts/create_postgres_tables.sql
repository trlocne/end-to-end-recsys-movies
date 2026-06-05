-- Create PostgreSQL tables for movie recommendation system

-- Table: items
CREATE TABLE IF NOT EXISTS public.items (
    item_id bigint PRIMARY KEY,
    title varchar,
    genre jsonb,
    tags jsonb,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);

-- Table: interactions
CREATE TABLE IF NOT EXISTS public.interactions (
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL,
    item_id bigint NOT NULL,
    rating double precision,
    timestamp timestamp without time zone,
    interaction_type varchar
);

-- Table: users
CREATE TABLE IF NOT EXISTS public.users (
    user_id bigint PRIMARY KEY,
    created_at timestamp without time zone,
    updated_at timestamp without time zone,
    metadata jsonb
);

-- ============================================================
-- Debezium CDC: replication slot + publication
-- Chay sau khi tao xong 3 tables tren
-- ============================================================

-- Tao replication slot cho Debezium (chi tao 1 lan)
SELECT pg_create_logical_replication_slot('debezium_docker', 'pgoutput')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_replication_slots WHERE slot_name = 'debezium_docker'
);

-- Tao publication cho 3 tables CDC
DROP PUBLICATION IF EXISTS dbz_publication;
CREATE PUBLICATION dbz_publication FOR TABLE public.interactions, public.users, public.items;

-- Verify
SELECT * FROM pg_publication WHERE pubname = 'dbz_publication';
SELECT * FROM pg_replication_slots WHERE slot_name = 'debezium_docker';