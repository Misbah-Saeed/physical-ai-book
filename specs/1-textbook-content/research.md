# Research Decisions for Physical AI & Humanoid Robotics Textbook Content

## Resolved NEEDS CLARIFICATION from Plan.md

### Language/Version
-   **Decision**: Use the latest stable versions for Node.js, Python, ROS 2, and C++ for examples.
-   **Rationale**: Opting for the latest stable versions ensures access to the newest features, performance improvements, and security updates. Specific versions will be pinned in `package.json` and `requirements.txt` during implementation.
-   **Alternatives considered**: Specific older versions for broader compatibility (rejected to leverage modern features).

### Primary Dependencies
-   **Decision**: Use the latest stable versions for Docusaurus, React, OpenAI API, FastAPI, Qdrant, and Neon Postgres.
-   **Rationale**: Similar to language versions, using the latest stable dependencies provides the best tooling and features. Specific versions will be managed via package managers.
-   **Alternatives considered**: Pinning to older versions (rejected for the same reasons as language versions).

### Testing Frameworks
-   **Decision**: Use Jest for Docusaurus components (React) and Pytest for FastAPI backend components.
-   **Rationale**: Jest is a widely adopted and feature-rich testing framework for JavaScript/React, and Pytest is a popular and flexible framework for Python. These choices align with best practices for their respective ecosystems.
-   **Alternatives considered**: React Testing Library, Mocha for frontend; unittest for Python (rejected for preferring the more modern and widely used Jest and Pytest).

### Performance Goals
-   **Decision**:
    -   Docusaurus site content rendering: Aim for Time To Interactive (TTI) under 2.5 seconds on a simulated mobile 3G network.
    -   Docusaurus site build times: Aim for full build completion under 5 minutes for initial content.
    -   RAG Chatbot responses: Target p95 latency under 2 seconds for typical queries.
    -   Advanced features (personalization, translation): Latency targets to be defined as these features are prioritized and detailed.
-   **Rationale**: These targets provide a balance between user experience and development effort for an educational platform. Specific metrics and monitoring will be established during implementation.
-   **Alternatives considered**: More aggressive performance targets (rejected to allow flexibility in initial content development), less specific targets (rejected for lack of measurability).