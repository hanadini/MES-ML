# Multi-stage Dockerfile to build the Maven project and run the Spring Boot JAR
# Location: src/main/resources/Dockerfile

# 1) Build stage: use Maven with Java 17 to build the fat jar
FROM maven:3.9.4-eclipse-temurin-17 AS builder
WORKDIR /workspace

# copy only maven files first for better layer caching
COPY pom.xml ./
COPY src ./src

# build the project (skip tests to speed up; remove -DskipTests if you want tests run)
RUN mvn -B -DskipTests package

# 2) Runtime stage: use a slim JRE image to run the produced jar
FROM eclipse-temurin:17-jre
WORKDIR /app

# copy the jar produced by the builder stage
COPY --from=builder /workspace/target/*.jar app.jar

# default port for Spring Boot; change if your app uses a different port
EXPOSE 8080

ENV JAVA_OPTS=""

# run the jar; using sh -c so JAVA_OPTS can be injected at runtime
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app/app.jar"]

