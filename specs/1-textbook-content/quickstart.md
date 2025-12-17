# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

This guide will help you set up and run the Docusaurus-based textbook locally.

## Prerequisites

Before you begin, ensure you have the following installed:

-   Node.js (latest LTS version recommended)
-   npm or Yarn (Yarn Classic recommended)
-   Git

## 1. Clone the Repository

First, clone the textbook repository to your local machine:

```bash
git clone [REPOSITORY_URL]
cd [REPOSITORY_NAME]
```

## 2. Install Dependencies

Navigate to the project root directory and install the Docusaurus dependencies:

```bash
yarn install
# or npm install
```

## 3. Start the Development Server

Once the dependencies are installed, you can start the Docusaurus development server:

```bash
yarn start
# or npm start
```

This will open a new browser window/tab at `http://localhost:3000` where you can view the textbook. Docusaurus features hot-reloading, so any changes you make to the Markdown files or React components will be reflected instantly.

## 4. Build for Production

To create a production-ready build of the textbook, run:

```bash
yarn build
# or npm run build
```

The static files will be generated in the `build/` directory, ready for deployment to platforms like GitHub Pages.

## Next Steps

-   Explore the `docs/` directory to see the chapter structure.
-   Customize the Docusaurus configuration in `docusaurus.config.js`.
-   Start adding your own content to new or existing Markdown files in `docs/`.

## RAG Chatbot Backend (Optional)

Setting up the RAG chatbot backend involves Python dependencies and database configuration. Refer to the `chatbot_backend/README.md` (once created) for detailed instructions.