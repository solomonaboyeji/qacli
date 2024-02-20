stop-database:
	docker compose -f supabase/docker/docker-compose.yml stop
	
clean-database:
	docker compose -f supabase/docker/docker-compose.yml down -v
	rm -rf volumes/db/data/

start-database:
	docker compose -f supabase/docker/docker-compose.yml up -d

restart-database:
	docker compose -f supabase/docker/docker-compose.yml restart