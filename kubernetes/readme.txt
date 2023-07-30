host: mypostgres
port: 5432
username: postgres
password: [value of mypostgres-secret/superUserPassword]

kubectl apply -f kubernetes_postgresql.yaml
kubectl apply -f postgres-secret.yaml
kubectl apply -f my-postgres.yaml

kubectl port-forward service/mypostgres 5432:5432

