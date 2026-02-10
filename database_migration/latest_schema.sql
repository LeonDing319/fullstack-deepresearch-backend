-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.research_results (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  session_id character varying,
  research_id character varying NOT NULL,
  model character varying NOT NULL,
  duration numeric NOT NULL,
  stage_timings jsonb NOT NULL,
  sources_found integer DEFAULT 0,
  word_count integer DEFAULT 0,
  success boolean NOT NULL,
  error text,
  report_content text NOT NULL,
  supervisor_tools_used ARRAY DEFAULT '{}'::text[],
  research_brief text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT research_results_pkey PRIMARY KEY (id),
  CONSTRAINT research_results_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.research_sessions(session_id)
);
CREATE TABLE public.research_sessions (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  session_id character varying NOT NULL UNIQUE,
  query text NOT NULL,
  session_type character varying NOT NULL CHECK (session_type::text = ANY (ARRAY['individual'::character varying, 'comparison'::character varying]::text[])),
  timestamp timestamp with time zone DEFAULT now(),
  user_feedback jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT research_sessions_pkey PRIMARY KEY (id)
);