/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SlashCommand } from './types.js';
import { CommandKind } from './types.js';
import { MessageType } from '../types.js';

export const anonymizeCommand: SlashCommand = {
  name: 'anonymize',
  description:
    'Enable or disable session-level secret anonymization. Default: on',
  kind: CommandKind.BUILT_IN,
  action: (context, args) => {
    const { config } = context.services;
    const { addItem } = context.ui;

    if (!config) {
      // This should not happen, but good to have a safeguard.
      addItem(
        {
          type: MessageType.ERROR,
          text: 'Unable to access configuration.',
        },
        Date.now(),
      );
      return;
    }

    const arg = args.trim().toLowerCase();

    if (arg === 'on') {
      config.setAnonymizationEnabled(true);
      addItem(
        {
          type: MessageType.INFO,
          text: 'Secret anonymization is now ON for this session.',
        },
        Date.now(),
      );
    } else if (arg === 'off') {
      config.setAnonymizationEnabled(false);
      addItem(
        {
          type: MessageType.INFO,
          text: 'Secret anonymization is now OFF for this session.',
        },
        Date.now(),
      );
    } else {
      const currentState = config.isAnonymizationEnabled() ? 'ON' : 'OFF';
      addItem(
        {
          type: MessageType.INFO,
          text: `Usage: /anonymize [on|off]\nAnonymization is currently ${currentState}.`,
        },
        Date.now(),
      );
    }
  },
};
