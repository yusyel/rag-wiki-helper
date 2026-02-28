FROM postgres:16-alpine

ENV POSTGRES_DB=ragdb
ENV POSTGRES_USER=raguser
ENV POSTGRES_PASSWORD=ragpass

COPY init.sql /docker-entrypoint-initdb.d/01-init.sql

EXPOSE 5432
