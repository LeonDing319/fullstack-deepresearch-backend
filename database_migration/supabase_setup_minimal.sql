-- Minimal Supabase Database Setup for Deep Research Agent
-- This is the clean, production-ready version with only essential tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE TABLES (Only 2 tables needed!)
-- ============================================================================

-- Research Sessions Table (handles both individual runs and comparisons)
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    query TEXT NOT NULL,
    session_type VARCHAR(20) NOT NULL CHECK (session_type IN ('individual', 'comparison')),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_feedback JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Research Results Table (stores all model runs)
CREATE TABLE IF NOT EXISTS research_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    research_id VARCHAR(255) NOT NULL, -- Individual research ID for tracking
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

CREATE INDEX IF NOT EXISTS idx_research_sessions_timestamp ON research_sessions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_research_sessions_session_id ON research_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_research_sessions_type ON research_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_research_results_session_id ON research_results(session_id);
CREATE INDEX IF NOT EXISTS idx_research_results_research_id ON research_results(research_id);
CREATE INDEX IF NOT EXISTS idx_research_results_model ON research_results(model);
CREATE INDEX IF NOT EXISTS idx_research_results_created_at ON research_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_results_success ON research_results(success);

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

CREATE TRIGGER update_research_sessions_updated_at 
    BEFORE UPDATE ON research_sessions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SECURITY (RLS) - Properly configured policies
-- ============================================================================

ALTER TABLE research_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_results ENABLE ROW LEVEL SECURITY;

-- Research Sessions Policies
CREATE POLICY "Enable read access for all users" ON research_sessions FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON research_sessions FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON research_sessions FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON research_sessions FOR DELETE USING (true);

-- Research Results Policies  
CREATE POLICY "Enable read access for all users" ON research_results FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON research_results FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON research_results FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON research_results FOR DELETE USING (true);

-- ============================================================================
-- ANALYTICS VIEWS (Only 3 views needed!)
-- ============================================================================

-- 1. Model Performance Summary (for Compare tab Performance Overview)
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model,
    COUNT(*) as total_requests,
    ROUND(AVG(duration)::NUMERIC, 2) as average_duration,
    ROUND((COUNT(CASE WHEN success THEN 1 END)::NUMERIC / COUNT(*)) * 100, 2) as success_rate,
    ROUND(AVG(sources_found)::NUMERIC, 1) as average_sources_found,
    ROUND(AVG(word_count)::NUMERIC, 0) as average_word_count,
    MAX(created_at) as last_used
FROM research_results 
GROUP BY model
ORDER BY total_requests DESC;

-- 2. Individual Research History (for Research tab)
CREATE OR REPLACE VIEW individual_research_history AS
SELECT 
    rs.session_id,
    rs.query,
    rs.timestamp,
    rr.research_id,
    rr.model,
    rr.duration,
    rr.success,
    rr.sources_found,
    rr.word_count,
    rr.error,
    rr.created_at
FROM research_sessions rs
JOIN research_results rr ON rs.session_id = rr.session_id
WHERE rs.session_type = 'individual'
ORDER BY rs.timestamp DESC;

-- 3. Comparison History (for Compare tab history)
CREATE OR REPLACE VIEW comparison_history AS
SELECT 
    rs.session_id,
    rs.query,
    rs.timestamp,
    rs.user_feedback,
    COUNT(rr.id) as models_compared,
    AVG(rr.duration) as avg_duration,
    MIN(rr.duration) as fastest_duration,
    MAX(rr.duration) as slowest_duration,
    COUNT(CASE WHEN rr.success THEN 1 END) as successful_runs,
    rs.created_at
FROM research_sessions rs
LEFT JOIN research_results rr ON rs.session_id = rr.session_id
WHERE rs.session_type = 'comparison'
GROUP BY rs.session_id, rs.query, rs.timestamp, rs.user_feedback, rs.created_at
ORDER BY rs.timestamp DESC;

-- ============================================================================
-- SETUP COMPLETE! 
-- Total: 2 tables + 3 views + indexes + security
-- ============================================================================
