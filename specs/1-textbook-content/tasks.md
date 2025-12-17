# Tasks: Physical AI & Humanoid Robotics Textbook Content

**Input**: Design documents from `/specs/1-textbook-content/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The feature specification did not explicitly request test tasks. However, the plan mentions "Docusaurus build validation, unit tests for Python backend components (pytest), integration tests for chatbot." I will include relevant test tasks as part of the implementation for completeness.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `docs/`, `src/`, `static/`, `personalization_backend/`, `localization_backend/` at repository root

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create base Docusaurus structure (root directory)
- [ ] T002 Configure Docusaurus `docusaurus.config.js`
- [ ] T003 Configure Docusaurus `sidebars.js`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Create `docs/` directory for chapters
- [ ] T005 Create `src/components/`, `src/pages/`, `src/theme/` directories for Docusaurus components
- [ ] T006 Create `static/` directory for assets
- [ ] T007 Create `personalization_backend/` directory structure with `app/`, `requirements.txt`, `tests/` (Optional - Bonus Feature)
- [ ] T008 Create `localization_backend/` directory structure with `app/`, `requirements.txt`, `tests/` (Optional - Bonus Feature)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Read Core Chapter Content (Priority: P1) 🎯 MVP

**Goal**: Students can read the core content of each chapter and subchapter to understand Physical AI and Humanoid Robotics concepts.

**Independent Test**: A user can navigate to any core chapter, read its content, view diagrams, and understand code examples independently.

### Implementation for User Story 1

- [ ] T009 [US1] Create `docs/chapter1/index.md` for "Introduction to Physical AI & Embodied Intelligence"
- [ ] T010 [US1] Create `docs/chapter2/index.md` for "Sensors & Perception Foundations"
- [ ] T011 [US1] Create `docs/chapter3/index.md` for "ROS 2 Fundamentals"
- [ ] T012 [US1] Create `docs/chapter4/index.md` for "URDF & Humanoid Modeling"
- [ ] T013 [US1] Create `docs/chapter5/index.md` for "Gazebo Simulation"
- [ ] T014 [US1] Create `docs/chapter6/index.md` for "Unity Visualization and Interaction"
- [ ] T015 [US1] Create `docs/chapter7/index.md` for "NVIDIA Isaac Sim & Isaac ROS"
- [ ] T016 [US1] Create `docs/chapter8/index.md` for "Vision-Language-Action Systems"
- [ ] T017 [US1] Add objectives, summaries, and initial textual content for all chapters in `docs/chapterN/index.md`
- [ ] T018 [US1] Integrate ROS 2 nodes, topics, services, actions examples into relevant chapters (e.g., `docs/chapter3/index.md`)
- [ ] T019 [US1] Add minimal publisher/subscriber example in `docs/chapter3/index.md`
- [ ] T020 [US1] Include URDF examples for humanoids in `docs/chapter4/index.md`
- [ ] T021 [US1] Integrate Gazebo sample scripts in `docs/chapter5/index.md`
- [ ] T022 [US1] Integrate Isaac Sim & Isaac ROS sample scripts in `docs/chapter7/index.md`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Explore Docusaurus Structure (Priority: P2)

**Goal**: Developers can explore the Docusaurus book folder structure to understand how to add new content or modify existing chapters.

**Independent Test**: A user can clone the repository, open the Docusaurus project, and identify the correct file locations for chapters, assets, and configuration.

### Implementation for User Story 2

- [ ] T023 [US2] Ensure `docusaurus.config.js` and `sidebars.js` are well-commented for developer understanding.
- [ ] T024 [US2] Create a simple `README.md` in the project root explaining the Docusaurus structure and how to add new chapters.
- [ ] T025 [US2] Validate that the generated chapter files adhere to a consistent naming and frontmatter convention (`docs/chapterN/index.md`).

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Utilize Advanced Features (Priority: P3)

**Goal**: Students can utilize optional advanced features like AI-assisted reading, personalized content, and Urdu translation to enhance their learning experience.

**Independent Test**:A user can view chapter content, access personalized content if available, and toggle the Urdu translation for a chapter.

### Implementation for Personalization Engine (Bonus)

- [ ] T026 [US3] Create Better-Auth signup page `src/pages/signup.js`
- [ ] T027 [US3] Create onboarding questionnaire component `src/components/OnboardingForm.js`
- [ ] T028 [US3] Implement saving user skills to Neon Postgres DB via `personalization_backend/app/db/`
- [ ] T029 [US3] Add personalization toggle button component `src/components/PersonalizationToggle.js`
- [ ] T030 [US3] Apply user profile filters to modify chapter content dynamically in `src/theme/DocItem.js`

### Implementation for Urdu Translation System (Bonus)

- [ ] T031 [US3] Add translation button component `src/components/TranslationButton.js`
- [ ] T032 [P] [US3] Create API route for translation in `localization_backend/app/api.py`
- [ ] T033 [P] [US3] Implement translation cache in Neon Postgres via `localization_backend/app/db/`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Build & Deploy

**Purpose**: Preparing the textbook for production and deployment.

- [ ] T034 Configure GitHub Pages deployment settings in `docusaurus.config.js`
- [ ] T035 Run Docusaurus build command (`yarn build`) and verify output in `build/`
- [ ] T036 Test deployment of the Docusaurus site on GitHub Pages

---

## Phase 7: Final Review (Polish & Cross-Cutting Concerns)

**Purpose**: Quality assurance and final checks across the entire textbook.

- [ ] T037 Fix all broken internal links within `docs/`
- [ ] T038 Fix all broken external links
- [ ] T039 Validate all code blocks for correctness and executability across chapters

---

## Dependencies & Execution Order

### Phase Dependencies

-   **Setup (Phase 1)**: No dependencies - can start immediately
-   **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
-   **User Stories (Phase 3-5)**: All depend on Foundational phase completion
    -   User stories can then proceed in parallel (if staffed)
    -   Or sequentially in priority order (P1 → P2 → P3)
-   **Build & Deploy (Phase 6)**: Depends on all desired user stories being complete.
-   **Final Review (Phase 7)**: Depends on Build & Deploy phase completion.

### User Story Dependencies

-   **User Story 1 (P1 - Read Core Chapter Content)**: Can start after Foundational (Phase 2) - No dependencies on other stories.
-   **User Story 2 (P2 - Explore Docusaurus Structure)**: Can start after Foundational (Phase 2) - No dependencies on other stories; relies on User Story 1 content for structure.

### Within Each User Story

-   Models/database setup before services/API endpoints.
-   Frontend components after backend API routes are defined.
-   Core implementation before integration.
-   Story complete before moving to next priority.

### Parallel Opportunities

-   All Setup tasks marked [P] can run in parallel.
-   All Foundational tasks marked [P] can run in parallel (within Phase 2).
-   Once Foundational phase completes, User Story 1 (content creation) can begin.
-   Tasks marked with [P] (e.g., creating multiple backend services or frontend components) can run in parallel.

---


## Implementation Strategy

### MVP First (User Story 1 & 2 Only)

1.  Complete Phase 1: Setup
2.  Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3.  Complete Phase 3: User Story 1 (Core Chapter Content)
4.  Complete Phase 4: User Story 2 (Explore Docusaurus Structure)
5.  **STOP and VALIDATE**: Test User Stories 1 & 2 independently.
6.  Deploy/demo if ready.

### Incremental Delivery

1.  Complete Setup + Foundational → Foundation ready
2.  Add User Story 1 (P1) → Test independently → Deploy/Demo
3.  Add User Story 2 (P2) → Test independently → Deploy/Demo
4.  Add User Story 3 (P3 - RAG Chatbot System) → Test independently → Deploy/Demo
5.  Add remaining bonus features (Personalization, Urdu Translation) incrementally.
6.  Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1.  Team completes Setup + Foundational together.
2.  Once Foundational is done:
    -   Developer A: User Story 1 (Content Creation)
    -   Developer B: User Story 2 (Docusaurus Structure Documentation & Validation)
3.  Stories complete and integrate independently.

---

## Notes

-   [P] tasks = different files, no dependencies
-   [Story] label maps task to specific user story for traceability
-   Each user story should be independently completable and testable
-   Verify tests fail before implementing (where applicable)
-   Commit after each task or logical group
-   Stop at any checkpoint to validate story independently
-   Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
