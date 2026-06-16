#!

# create session
cd /eos/user/m/mamozzan || exit

# once you do the command, go to the link and copy the code
kubectl create -f session.yaml

# check the availability of the pod
kubectl get po

# connect
ssh session-1@ngt.cern.ch

# deleting a session
kubectl delete po session-1
