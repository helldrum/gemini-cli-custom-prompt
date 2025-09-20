/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2-0
 */

// This file contains custom redaction rules adapted from the gitleaks project.
// See: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml

interface RedactionRule {
  name: string;
  pattern: () => RegExp;
}

// Using a function to get the regex so that we get a new instance each time.
// This is important because the 'g' flag makes the regex stateful.
const gitleaksRules: RedactionRule[] = [
  {
    name: 'Private Key',
    pattern: () =>
      /(-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY-----[\s\S]*?-----END[ A-Z0-9_-]{0,100}PRIVATE KEY-----\s*\n?)/g,
  },
  {
    // Source: Adapted from various gitleaks rules for generic secrets.
    name: 'Generic Secret',
    pattern: () =>
      /(secret|token|password|bearer|key|auth)[_A-Z0-9]*\s*[:=]\s*['"]?[^,\n'"]+['"]?/gi,
  },
  {
    name: 'JDBC Connection String',
    pattern: () =>
      /jdbc:[a-zA-Z0-9]+:\/\/[^\s'"]+(?:user|username|password|passwd|pwd)=[^&'"]+/gi,
  },
  {
    name: 'Redis Connection String',
    pattern: () => /redis(s)?:\/\/(?:[^:\s]+(?::[^@\s]+)?@)?[^\s'\"]+/gi,
  },
  {
    name: 'WebSocket Secure Connection String',
    pattern: () => /wss:\/\/[^:\s]+:[^@\s]+@[^\s'\"]+/gi,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L72
    // This regex uses negative lookarounds to reduce false positives.
    name: 'AWS Secret Key',
    pattern: () => /(?<![A-Za-z0-9/+=])([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])/g,
  },
  {
    name: 'Phone Number',
    pattern: () =>
      /\b(?:\+?\d{1,3}[-. ]\s?)?(?:\(?\d{2,4}\)?[-. ]?){2,}\d{2,4}\b|\b\d{2}(?:\.\d{2}){4}\b/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L63
    // The regex is enhanced based on https://awsteele.com/blog/2020/09/26/aws-access-key-format.html
    // to be more specific for AKIA keys.
    name: 'AWS Access Key',
    pattern: () =>
      /AKIA[A-Z2-7]{16}|(A3T[A-Z0-9]|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}/g,
  },
  {
    name: 'Email Address',
    pattern: () => /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L998
    name: 'GitHub Fine-Grained PAT',
    pattern: () => /github_pat_\w{82}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L1010
    name: 'GitHub PAT',
    pattern: () => /ghp_[0-9a-zA-Z]{36}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L1002
    name: 'GitHub OAuth',
    pattern: () => /gho_[0-9a-zA-Z]{36}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L994
    name: 'GitHub App Token',
    pattern: () => /(?:ghu|ghs)_[0-9a-zA-Z]{36}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L595
    name: 'GCP API Key',
    pattern: () => /AIza[\w-]{35}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L1206
    name: 'Twilio API Key',
    pattern: () => /SK[0-9a-fA-F]{32}/g,
  },
  {
    // Source: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml#L1154
    name: 'Stripe Access Token',
    pattern: () => /(?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{10,99}/g,
  },
];

export const customRules = gitleaksRules;
