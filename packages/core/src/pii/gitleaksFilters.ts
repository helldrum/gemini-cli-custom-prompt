/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

interface RedactionRule {
  name: string;
  pattern: () => RegExp;
}

const gitleaksRules: RedactionRule[] = [
  {
    name: 'GCP Service Account',
    pattern: () =>
      /{\s*"type":\s*"service_account"[\s\S]*"private_key":[\s\S]*?}/g,
  },
  {
    name: 'Private Key',
    pattern: () =>
      /(-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY-----[\s\S]*?-----END[ A-Z0-9_-]{0,100}PRIVATE KEY-----)/g,
  },
  {
    name: 'SSL Certificate',
    pattern: () =>
      /(-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----)/g,
  },
  {
    name: 'MySQL Credentials',
    pattern: () => /mysql:\/\/[!-~]*/g,
  },
  {
    name: 'PostgreSQL Credentials',
    pattern: () => /postgres(ql)?:\/\/[!-~]*/g,
  },
  {
    name: 'MongoDB Credentials',
    pattern: () => /mongodb(?:\+srv)?:\/\/[!-~]*/g,
  },
  {
    name: 'Redis Credentials',
    pattern: () => /redis:\/\/[!-~]*/g,
  },
  {
    name: 'MS SQL Credentials',
    pattern: () => /(?:mssql|sqlserver):\/\/[!-~]*/g,
  },
  {
    name: 'Oracle Credentials',
    pattern: () => /jdbc:oracle:thin:[!-~]*/g,
  },
  {
    name: 'Cassandra Credentials',
    pattern: () => /cassandra:\/\/[!-~]*/g,
  },
  {
    name: 'SQLlite Credentials',
    pattern: () => /sqlite:\/\/[!-~]*/g,
  },
  {
    name: 'Stripe Access Token',
    pattern: () => /(?:pk|sk|rk)_(?:test|live|prod)_[!-~]*/g,
  },
  {
    name: 'Email Address',
    pattern: () => /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
  },
];

export const customRules = gitleaksRules;
