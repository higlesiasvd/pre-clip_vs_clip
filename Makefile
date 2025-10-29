.PHONY: build run-preclip run-clip run-compare run-all shell clean help local-all

help:
	@echo "=========================================="
	@echo "Práctica 2: Pre-CLIP y CLIP"
	@echo "=========================================="
	@echo ""
	@echo "Comandos DOCKER:"
	@echo "  make build          - Construir imagen Docker"
	@echo "  make run-preclip    - Ejecutar tarea Pre-CLIP"
	@echo "  make run-clip       - Ejecutar tarea CLIP"
	@echo "  make run-compare    - Comparar resultados"
	@echo "  make run-all        - Ejecutar todo en secuencia"
	@echo "  make shell          - Acceder a shell interactiva"
	@echo "  make clean          - Limpiar contenedores y caché"
	@echo ""
	@echo "Comandos LOCALES (sin Docker):"
	@echo "  make local-all      - Ejecutar todo localmente"
	@echo ""
	@echo "Requisitos:"
	@echo "  - Docker: Docker y Docker Compose instalados"
	@echo "  - Local: pip install -r requirements.txt"
	@echo "  - Carpeta 'dataset/' con imágenes y metadata.json"
	@echo ""

# ============ COMANDOS LOCALES (SIN DOCKER) ============

local-all:
	@echo "Ejecutando localmente (sin Docker)..."
	@echo ""
	@if [ ! -d ".venv" ]; then \
		echo "❌ Error: No encontrado .venv"; \
		echo ""; \
		echo "Ejecuta:"; \
		echo "  python3 -m venv .venv"; \
		echo "  source .venv/bin/activate"; \
		echo "  pip install -r requirements.txt"; \
		exit 1; \
	fi
	@echo "Activando .venv y ejecutando..."
	@bash -c "source .venv/bin/activate && python3 preclip.py"
	@echo ""
	@bash -c "source .venv/bin/activate && python3 clip_task.py"
	@echo ""
	@bash -c "source .venv/bin/activate && python3 compare.py"
	@echo ""
	@echo "Pipeline completado"

# ============ COMANDOS DOCKER ============

build:
	@echo "Construyendo imagen Docker..."
	docker compose build
	@echo "Imagen construida"

run-preclip:
	@echo "Ejecutando Pre-CLIP..."
	docker compose run --rm practica2 python3 preclip.py

run-clip:
	@echo "Ejecutando CLIP..."
	docker compose run --rm practica2 python3 clip_task.py

run-compare:
	@echo "Comparando resultados..."
	docker compose run --rm practica2 python3 compare.py

run-all:
	@echo "Ejecutando pipeline completo..."
	@echo ""
	@$(MAKE) run-preclip
	@echo ""
	@$(MAKE) run-clip
	@echo ""
	@$(MAKE) run-compare
	@echo ""
	@echo "Pipeline completado"

shell:
	@echo "Accediendo a shell interactiva..."
	docker compose run --rm practica2 /bin/bash

clean:
	@echo "Limpiando contenedores y caché..."
	docker compose down -v
	docker rmi practica2-clip 2>/dev/null || true
	@echo "Limpieza completada"

# Alias para compatibilidad
start: run-all
stop: clean