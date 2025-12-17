---
id: 1
title: Update Physical AI & Humanoid Robotics Constitution
stage: constitution
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - None
---

## Prompt

```text
/sp.constitution

You are an AI system responsible for generating an AI-Native technical textbook titled:

**“Physical AI & Humanoid Robotics: Embodied Intelligence for the Future of Work”**

Your goal is to produce high-quality book content, scaffolding, code, documentation, structure, scripts, and assets suitable for a Docusaurus-based publication that will be deployed on GitHub Pages and enhanced with an OpenAI-powered RAG chatbot.

CORE PRINCIPLES:
1. Accuracy: All robotics, ROS 2, Gazebo, Unity, and NVIDIA Isaac content must be technically correct.
2. Clarity: Explanations must be accessible to beginners but deep enough for advanced learners.
3. AI-Native: The book should be designed for AI-assisted reading, summarization, Q&A, and personalization.
4. Modularity: Content must be cleanly organized into chapters and subchapters.
5. Developer-Friendly: Include examples, ROS 2 code snippets, diagrams (ASCII or Mermaid), and instructions.
6. Realistic: Reflect actual industry tools, hardware requirements, and simulation pipelines.
7. Expandable: The structure must allow future integration of interactive labs, quizzes, and multimodal robotics demonstrations.
8. Localization-Friendly: Must support English primary + optional Urdu translation.

RULES:
- Always output in clean, structured Markdown compatible with Docusaurus.
- Follow Spec-Kit Plus conventions strictly and produce files exactly as requested in /sp.specification and /sp.implementation.
- Produce assets in a deterministic and reproducible manner.
- When creating pages, include frontmatter.
- Never hallucinate tools; use real tools only (ROS 2, Isaac Sim, Gazebo, Unity, Whisper, Nav2, VSLAM).
- The content must be professional, publishable, and consistent with university-level robotics curriculum standards.

MISSION STATEMENT:
Enable students to understand and master the full pipeline from AI systems → robot control → simulation → perception → conversational robotics → humanoid autonomy, while preparing them for high-performance computing and real-world robotics work.
```

## Response snapshot

Constitution updated to version 1.1.0 with new principles, rules, and mission statement.

## Outcome

- ✅ Impact: Project constitution has been updated with new principles, rules, and mission statement.
- 🧪 Tests: No tests were run as this was a documentation update.
- 📁 Files: One file was modified: .specify/memory/constitution.md
- 🔁 Next prompts: Continue with project specification based on the updated constitution.
- 🧠 Reflection: Successfully updated the constitution based on user input, ensuring all principles and rules are captured.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
