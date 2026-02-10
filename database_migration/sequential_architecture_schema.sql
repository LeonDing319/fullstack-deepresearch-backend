-- Sequential Research Architecture - Database Schema
-- Optimized for one-at-a-time research with historical comparisons

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE TABLES - Optimized for Sequential Architecture
-- ============================================================================

-- Research Sessions Table (simplified for sequential approach)
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    query TEXT NOT NULL,
    session_type VARCHAR(20) NOT NULL DEFAULT 'individual' CHECK (session_type IN ('individual', 'comparison')),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_feedback JSONB, -- For future rating features
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Research Results Table (one record per model run)
CREATE TABLE IF NOT EXISTS research_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    research_id VARCHAR(255) UNIQUE NOT NULL, -- Each research run has unique ID
    model VARCHAR(50) NOT NULL,
    duration DECIMAL(10,3) NOT NULL,
    stage_timings JSONB NOT NULL,
    sources_found INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    success BOOLEAN NOT NULL,
    error TEXT,
    report_content TEXT NOT NULL,
    supervisor_tools_used TEXT[] DEFAULT '{}',
    research_brief TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- Research Sessions
CREATE INDEX IF NOT EXISTS idx_research_sessions_timestamp ON research_sessions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_research_sessions_query ON research_sessions USING gin(to_tsvector('english', query));

-- Research Results  
CREATE INDEX IF NOT EXISTS idx_research_results_created_at ON research_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_results_model ON research_results(model);
CREATE INDEX IF NOT EXISTS idx_research_results_success ON research_results(success);
CREATE INDEX IF NOT EXISTS idx_research_results_duration ON research_results(duration);
CREATE INDEX IF NOT EXISTS idx_research_results_research_id ON research_results(research_id);

-- ============================================================================
-- AUTO-UPDATE TRIGGERS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_research_sessions_updated_at ON research_sessions;
CREATE TRIGGER update_research_sessions_updated_at 
    BEFORE UPDATE ON research_sessions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SECURITY (RLS)
-- ============================================================================

ALTER TABLE research_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_results ENABLE ROW LEVEL SECURITY;

-- Research Sessions Policies (drop existing first to avoid conflicts)
DROP POLICY IF EXISTS "Enable read access for all users" ON research_sessions;
DROP POLICY IF EXISTS "Enable insert access for all users" ON research_sessions;
DROP POLICY IF EXISTS "Enable update access for all users" ON research_sessions;
DROP POLICY IF EXISTS "Allow all operations on research_sessions" ON research_sessions;

CREATE POLICY "Enable read access for all users" ON research_sessions FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON research_sessions FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON research_sessions FOR UPDATE USING (true);

-- Research Results Policies (drop existing first to avoid conflicts)
DROP POLICY IF EXISTS "Enable read access for all users" ON research_results;
DROP POLICY IF EXISTS "Enable insert access for all users" ON research_results;
DROP POLICY IF EXISTS "Enable update access for all users" ON research_results;
DROP POLICY IF EXISTS "Allow all operations on research_results" ON research_results;

CREATE POLICY "Enable read access for all users" ON research_results FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON research_results FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON research_results FOR UPDATE USING (true);

-- ============================================================================
-- ANALYTICS VIEWS - Optimized for Sequential Architecture
-- ============================================================================

-- 1. Research History View (for Compare tab and History tab)
CREATE OR REPLACE VIEW research_history AS
SELECT 
    rs.session_id,
    rs.query,
    rs.timestamp as session_timestamp,
    rr.research_id,
    rr.model,
    rr.duration,
    rr.success,
    rr.sources_found,
    rr.word_count,
    rr.error,
    rr.stage_timings,
    rr.supervisor_tools_used,
    rr.created_at,
    -- Quality score calculation
    CASE 
        WHEN rr.success THEN 
            LEAST(100, (rr.sources_found * 10) + (rr.word_count / 10))
        ELSE 0 
    END as quality_score
FROM research_sessions rs
JOIN research_results rr ON rs.session_id = rr.session_id
ORDER BY rr.created_at DESC;

-- 2. Model Performance Summary (for Performance Overview)
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model,
    COUNT(*) as total_requests,
    ROUND(AVG(duration)::NUMERIC, 2) as average_duration,
    ROUND((COUNT(CASE WHEN success THEN 1 END)::NUMERIC / COUNT(*)) * 100, 2) as success_rate,
    ROUND(AVG(sources_found)::NUMERIC, 1) as average_sources_found,
    ROUND(AVG(word_count)::NUMERIC, 0) as average_word_count,
    MIN(duration) as fastest_duration,
    MAX(duration) as slowest_duration,
    MAX(created_at) as last_used
FROM research_results 
GROUP BY model
ORDER BY total_requests DESC;

-- 3. Recent Research Summary (for dashboard widgets)
CREATE OR REPLACE VIEW recent_research_summary AS
SELECT 
    COUNT(*) as total_research_count,
    COUNT(CASE WHEN success THEN 1 END) as successful_count,
    COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as recent_count,
    ROUND(AVG(duration)::NUMERIC, 2) as average_duration,
    STRING_AGG(DISTINCT model, ', ') as models_used
FROM research_results;

-- ============================================================================
-- SAMPLE QUERIES for Frontend Integration
-- ============================================================================

/*
-- Get research history for Compare tab
SELECT * FROM research_history 
WHERE success = true 
ORDER BY created_at DESC 
LIMIT 50;

-- Compare specific research results
SELECT * FROM research_history 
WHERE research_id IN ('research_123', 'research_456', 'research_789');

-- Get model performance metrics
SELECT * FROM model_performance_summary;

-- Search research by query
SELECT * FROM research_history 
WHERE query ILIKE '%AI%' 
ORDER BY created_at DESC;

-- Get research by model
SELECT * FROM research_history 
WHERE model = 'openai' 
ORDER BY created_at DESC 
LIMIT 20;
*/
