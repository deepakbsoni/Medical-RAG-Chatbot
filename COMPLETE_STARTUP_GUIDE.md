# 🚀 Complete RAG System Startup Guide
## Healthcare - With CORS Proxy Integration

## 🎯 **Complete Service Architecture**
```
Medical Frontend (3000) → RAG Server (8002) → CORS Proxy (8001) → SSH Tunnel (8000) → Remote LLM
```

## 📋 **Step-by-Step Startup Sequence**

### **Terminal 1: SSH Tunnel (Port 8000)**
```bash
cd "/Users/deepsoni/Oracle Content - Accounts/Oracle Content/Projects/CoE_initial/CoE_base/000-AICoE-Knowledgebase/001-customers/Healthcare"

# Start SSH tunnel to remote LLM
ssh -i ./ssh-key-2023-08-03.key -L 127.0.0.1:8000:10.0.0.93:8000 opc@152.70.40.1

# Keep this terminal open - provides port 8000
```

### **Terminal 2: CORS Proxy (Port 8001)**
```bash
cd "/Users/deepsoni/Oracle Content - Accounts/Oracle Content/Projects/CoE_initial/CoE_base/000-AICoE-Knowledgebase/001-customers/Healthcare"

# Start CORS proxy
python3 cors-proxy.py

# Should show: "CORS proxy running on http://localhost:8001"
```

### **Terminal 3: Enhanced RAG System (Ports 8002 & 3000)**
```bash
cd "/Users/deepsoni/Oracle Content - Accounts/Oracle Content/Projects/CoE_initial/CoE_base/000-AICoE-Knowledgebase/001-customers/Healthcare/RAG"

# Start enhanced RAG system
./scripts/start_rag_system.sh

# Should detect CORS proxy and start successfully
```

## ✅ **Verification Steps**

### **Check All Services Running:**
```bash
# Should show all 4 services:
lsof -i :3000,:8000,:8001,:8002

# Port 8000: SSH tunnel to remote LLM
# Port 8001: CORS proxy 
# Port 8002: RAG enhancement server
# Port 3000: Enhanced frontend
```

### **Test Each Layer:**
```bash
# Test SSH tunnel
curl http://localhost:8000/health

# Test CORS proxy  
curl http://localhost:8001/health

# Test RAG server
curl http://localhost:8002/health

# Test frontend
curl http://localhost:3000
```

## 🌐 **Access Enhanced Chatbot**
Open browser: **http://localhost:3000/enhanced-medical-chatbot.html**

## 🎯 **Expected Behavior**

### **Connection Status:**
- Green indicator in header
- "RAG Enhanced" badge visible
- Context panel shows "connected" status
- No connection error messages

### **Enhanced Conversation Example:**
```
User: "I have chest pain"
RAG Analysis: Symptom detected, urgency=critical
Enhanced Response: "I understand you're experiencing chest pain. This is concerning. When did this start? Does it radiate to your arm, jaw, or back?"

User: "Started 2 hours ago, radiating to left arm"  
RAG Analysis: Previous context + new info = MI pattern
Enhanced Response: "These symptoms together are very concerning for a possible heart attack. Please call emergency services immediately..."
```

## 🛠️ **Troubleshooting**

### **Issue: "Connection failed: Failed to fetch"**
**Cause:** One of the services in the chain is not running

**Debug:**
```bash
# Check each service individually
curl http://localhost:8000/health  # SSH tunnel
curl http://localhost:8001/health  # CORS proxy
curl http://localhost:8002/health  # RAG server
curl http://localhost:3000         # Frontend
```

**Solution:** Start the missing service(s)

### **Issue: Port conflicts**
```bash
# Stop all conflicting processes
./scripts/stop_rag_system.sh

# Kill any remaining processes
lsof -ti:3000 | xargs kill -9
lsof -ti:8001 | xargs kill -9  
lsof -ti:8002 | xargs kill -9

# Restart in correct order
```

### **Issue: CORS proxy not connecting to SSH tunnel**
- Verify SSH tunnel is running and responsive
- Check if remote LLM server is actually running
- Test direct connection: `curl http://localhost:8000`

## 🎉 **Success Indicators**

✅ **All 4 services running on correct ports**  
✅ **Enhanced chatbot loads without errors**  
✅ **Context panel shows real-time analysis**  
✅ **Conversations have memory and continuity**  
✅ **Symptom detection and urgency assessment working**  

Your Enhanced Medical RAG Chatbot with full conversation memory and context awareness is now ready! 🚀