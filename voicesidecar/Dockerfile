FROM node:25-slim

WORKDIR /app
# Dependencies first: this layer is cached unless the lockfile changes.
# opusscript (WebAssembly) rather than the native @discordjs/opus, so there is
# no compiler toolchain and no install script to run.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force
COPY src ./src

ENV NODE_ENV=production
EXPOSE 8765
USER node
CMD ["node", "src/index.js"]
