FROM node:24.22.2-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run check && npm run build

# The gateway serves the built UI; compose mounts its config (compose/nginx/default.conf).
FROM nginx:1.30.6-alpine
COPY --from=build /app/dist /usr/share/nginx/html
