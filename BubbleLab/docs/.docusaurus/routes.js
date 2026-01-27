import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/markdown-page',
    component: ComponentCreator('/markdown-page', '3d7'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', 'e5f'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', '9ca'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', 'b3d'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '74e'),
            routes: [
              {
                path: '/bubbles/overview',
                component: ComponentCreator('/bubbles/overview', '264'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/agi-inc-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/agi-inc-bubble', 'd89'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/ai-agent-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/ai-agent-bubble', '994'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/airtable-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/airtable-bubble', 'e3f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/apify-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/apify-bubble', '0ca'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/eleven-labs-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/eleven-labs-bubble', '610'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/followupboss-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/followupboss-bubble', '7cf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/github-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/github-bubble', '3ba'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/gmail-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/gmail-bubble', '2a0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/google-calendar-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/google-calendar-bubble', '66e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/google-drive-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/google-drive-bubble', 'bfd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/google-sheets-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/google-sheets-bubble', 'c0d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/hello-world-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/hello-world-bubble', 'c4c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/http-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/http-bubble', '967'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/postgresql-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/postgresql-bubble', '56c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/resend-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/resend-bubble', 'cd4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/slack-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/slack-bubble', '7fc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/slack-formatter-agent-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/slack-formatter-agent-bubble', '967'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/storage-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/storage-bubble', '385'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/service-bubbles/telegram-bubble',
                component: ComponentCreator('/bubbles/service-bubbles/telegram-bubble', '937'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/bubbleflow-validation-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/bubbleflow-validation-tool', '185'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/code-edit-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/code-edit-tool', '10f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/get-bubble-details-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/get-bubble-details-tool', '05c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/instagram-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/instagram-tool', '60b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/linkedin-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/linkedin-tool', 'a62'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/list-bubbles-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/list-bubbles-tool', '24c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/reddit-scrape-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/reddit-scrape-tool', 'd13'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/research-agent-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/research-agent-tool', '2f6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/sql-query-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/sql-query-tool', '4b2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/web-crawl-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/web-crawl-tool', '3d8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/web-extract-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/web-extract-tool', 'e06'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/web-scrape-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/web-scrape-tool', 'd50'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/web-search-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/web-search-tool', '37c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/bubbles/tool-bubbles/youtube-tool',
                component: ComponentCreator('/bubbles/tool-bubbles/youtube-tool', 'cf6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro',
                component: ComponentCreator('/intro', '9fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/key-concepts/bubble-lab-vs-mcps',
                component: ComponentCreator('/key-concepts/bubble-lab-vs-mcps', 'a98'),
                exact: true
              },
              {
                path: '/key-concepts/bubbles',
                component: ComponentCreator('/key-concepts/bubbles', '18c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/key-concepts/execution-pipeline',
                component: ComponentCreator('/key-concepts/execution-pipeline', 'b6a'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
