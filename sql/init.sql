-- ============================================================
-- RecSys Database Schema
-- Target: PostgreSQL (Aiven Cloud)
-- ============================================================

-- ------------------------------------------------------------
-- users
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    user_id    BIGINT    NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata   JSONB,
    CONSTRAINT users_pkey PRIMARY KEY (user_id)
);

-- ------------------------------------------------------------
-- items
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.items (
    item_id    BIGINT    NOT NULL,
    title      VARCHAR,
    genre      JSONB,
    tags       JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT items_pkey PRIMARY KEY (item_id)
);

-- ------------------------------------------------------------
-- interactions
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.interactions (
    id               BIGSERIAL        NOT NULL,
    user_id          BIGINT           NOT NULL,
    item_id          BIGINT           NOT NULL,
    rating           DOUBLE PRECISION,
    timestamp        TIMESTAMP        DEFAULT CURRENT_TIMESTAMP,
    interaction_type VARCHAR          DEFAULT 'rating',
    CONSTRAINT interactions_pkey          PRIMARY KEY (id),
    CONSTRAINT fk_interactions_user       FOREIGN KEY (user_id)  REFERENCES public.users(user_id)  ON DELETE CASCADE,
    CONSTRAINT fk_interactions_item       FOREIGN KEY (item_id)  REFERENCES public.items(item_id)  ON DELETE CASCADE,
    CONSTRAINT chk_interactions_rating    CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5))
);

CREATE INDEX IF NOT EXISTS idx_interactions_user_item
    ON public.interactions (user_id, item_id);

CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
    ON public.interactions (timestamp);

-- ------------------------------------------------------------
-- incremental_state
-- Tracks the latest processed timestamp per pipeline component.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.incremental_state (
    component  VARCHAR   NOT NULL,
    timestamp  TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT incremental_state_pkey PRIMARY KEY (component)
);

CREATE INDEX IF NOT EXISTS idx_incremental_state_component
    ON public.incremental_state (component);
