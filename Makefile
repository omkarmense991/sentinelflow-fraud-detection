# =========================
# Production
# =========================

prod-up:
	docker compose -f docker-compose.prod.yml up --build -d

prod-down:
	docker compose -f docker-compose.prod.yml down

prod-restart:
	docker compose -f docker-compose.prod.yml down
	docker compose -f docker-compose.prod.yml up --build -d

prod-logs:
	docker compose -f docker-compose.prod.yml logs -f

prod-logs-api:
	docker compose -f docker-compose.prod.yml logs -f api

prod-logs-db:
	docker compose -f docker-compose.prod.yml logs -f postgres

prod-ps:
	docker compose -f docker-compose.prod.yml ps

# =========================
# Development
# =========================

dev-up:
	docker compose -f docker-compose.dev.yml up --build -d

dev-down:
	docker compose -f docker-compose.dev.yml down

dev-restart:
	docker compose -f docker-compose.dev.yml down
	docker compose -f docker-compose.dev.yml up --build -d

dev-logs:
	docker compose -f docker-compose.dev.yml logs -f

dev-ps:
	docker compose -f docker-compose.dev.yml ps

dev-logs:
	docker compose -f docker-compose.dev.yml logs -f

dev-logs-api:
	docker compose -f docker-compose.dev.yml logs -f api

dev-logs-db:
	docker compose -f docker-compose.dev.yml logs -f postgres

# =========================
# Database
# =========================

db:
	docker exec -it fraud-postgres psql -U fraud_user -d fraud_detection