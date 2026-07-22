import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, '..', 'install.json');
const target = resolve(here, '..', '..', 'labextension', 'install.json');

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
