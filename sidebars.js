/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Introduction to Physical AI & Embodied Intelligence',
      link: {
        type: 'doc',
        id: 'chapter1/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'Sensors & Perception Foundations',
      link: {
        type: 'doc',
        id: 'chapter2/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      link: {
        type: 'doc',
        id: 'chapter3/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'URDF & Humanoid Modeling',
      link: {
        type: 'doc',
        id: 'chapter4/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'Gazebo Simulation',
      link: {
        type: 'doc',
        id: 'chapter5/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'Unity Visualization & Interaction',
      link: {
        type: 'doc',
        id: 'chapter6/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac Sim & Isaac ROS',
      link: {
        type: 'doc',
        id: 'chapter7/index',
      },
      items: [],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action (VLA) Systems',
      link: {
        type: 'doc',
        id: 'chapter8/index',
      },
      items: [],
    },
  ],
};

module.exports = sidebars;
