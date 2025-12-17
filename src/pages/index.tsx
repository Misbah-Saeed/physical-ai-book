import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header style={{padding: '4rem 0', textAlign: 'center', backgroundColor: '#282c34', color: 'white'}}>
      <div>
        <Heading as="h1" style={{fontSize: '3rem', marginBottom: '1rem'}}>
          {siteConfig.title}
        </Heading>
        <p style={{fontSize: '1.5rem', marginBottom: '2rem'}}>
          {siteConfig.tagline}
        </p>
        <div>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
            style={{textDecoration: 'none'}}
          >
            Docusaurus Tutorial - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />"
    >
      <HomepageHeader />

      <main>
        <section style={{padding: '2rem', textAlign: 'center'}}>
          <h1>Physical AI &amp; Embodied Intelligence</h1>
          <h2>
            Welcome to the textbook on <strong>Physical AI</strong> and <strong>Embodied Intelligence</strong>.
          </h2>
          <p>➡️ Use the sidebar to start learning 🚀</p>
        </section>

        <HomepageFeatures />
      </main>
    </Layout>
  );
}
