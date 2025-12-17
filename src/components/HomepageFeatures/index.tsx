import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';

type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Easy to Use',
    description: (
      <>
        This platform is designed to be simple, clear, and beginner-friendly.
      </>
    ),
  },
  {
    title: 'Focused Learning',
    description: (
      <>
        Learn Physical AI and Embodied Intelligence step-by-step using the sidebar.
      </>
    ),
  },
  {
    title: 'Future-Ready',
    description: (
      <>
        Covers modern topics like robotics, AI systems, and intelligent agents.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section style={{padding: '2rem 0'}}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
