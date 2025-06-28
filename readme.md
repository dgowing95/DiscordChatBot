# DiscordChatBot
This is a python application that runs a discord bot that responds using AI to user messages. The application is primarily designed to run on Kubernetes and has a helm chart available.

Ollama is used for the local AI and Redis is used as a database and memory store

## Requirements for installing
- A Kubernetes cluster (microk8s, k3s, minikube also work).
- Helm installed
- An Nvidia GPU with the Nvidia Container Runtime Toolkit installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A registered Discord bot and a valid Discord Bot Token for it

## Installation
1. Create your own values.yaml file and edit as necessary (You must provide a discord token in this file). The base/defaults can be found in charts/dis-ai-bot/values.yaml in this repository.
1. Install via Helm e.g. `helm upgrade --install "https://github.com/dgowing95/DiscordChatBot/releases/download/v1.20.0/dchatbot-v1.20.0.tgz" -f values.yaml --namespace `dchatbot` --create-namespace`
