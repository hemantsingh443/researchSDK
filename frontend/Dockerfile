# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies including Tailwind CSS and PostCSS
RUN npm install && \
    npm install -D tailwindcss@latest postcss@latest autoprefixer@latest postcss-nesting @tailwindcss/postcss

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config (from the build context)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
