# GKE ML Training Framework

A scalable machine learning training framework that supports both CPU and GPU training workloads on Google Kubernetes Engine (GKE). This framework provides a flexible architecture for training ML models with efficient resource utilization and easy deployment.

## 🚀 Features

- Supports both CPU and GPU training workflows
- Scalable Kubernetes-based deployment
- Efficient resource management
- Persistent storage for model artifacts and training data
- Distributed training support
- Modular and extensible architecture

## 📋 Prerequisites

- Google Cloud Platform (GCP) account with enabled APIs
- Google Kubernetes Engine (GKE) cluster
- `gcloud` CLI installed and configured
- `kubectl` configured to work with your GKE cluster
- Docker installed (for building custom images)
- Python 3.8+
- TensorFlow 2.x

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gke-ml-training.git
cd gke-ml-training
```

2. Create GKE cluster with terraform:
```bash
terraform init
terraform plan -var-file="prod-variables-file.tfvars" "-lock=false" 
terraform apply -var-file="prod-variables-file.tfvars" "-lock=false"
terraform destroy -var-file="prod-variables-file.tfvars" "-lock=false" 

```

3. Install NVIDIA device plugin:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

4. Build and push the Docker image to artifact registry:

## 🎮 Usage

### Deploy Storage Resources

```bash
kubectl apply -f k8s/persistent-volume-claims.yaml
```

### Running Training Jobs

1. CPU Training:
```bash
kubectl apply -f k8s/cpu-training-job.yaml
```

2. GPU Training:
```bash
kubectl apply -f k8s/gpu-training-job.yaml
```

### Monitoring Training Progress

Monitor job status:
```bash
kubectl get jobs
```

Check training logs:
```bash
# For CPU training
kubectl logs -f job/ml-training-cpu

# For GPU training
kubectl logs -f job/ml-training-gpu
```


## 🐛 Troubleshooting

### Common Issues

1. GPU jobs stuck in "Pending":
- Verify GPU nodes are available
- Check NVIDIA device plugin status
- Verify resource quotas

2. Slow GPU Training:
- Check GPU utilization with `nvidia-smi`
- Verify batch size is optimized for GPU
- Check data pipeline efficiency

### Debugging Commands

```bash
# Check pod status
kubectl describe pod <pod-name>

# Check GPU status
kubectl exec -it <pod-name> -- nvidia-smi

# Check resource usage
kubectl top pods
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Google Cloud Platform
- Kubernetes community
- TensorFlow team
- NVIDIA for GPU support

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.
