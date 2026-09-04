# Quick Start

## Connect to server

```bash
ssh abedin@pascal.l3s.intra
# or, if configured: ssh pascal
```

## Once connected

```bash
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

python -c "import config; config.print_current_config()"
```

## Docs

See `REMOTE_SETUP_GUIDE.md` for SSH setup and full instructions.

**Server:** `pascal.l3s.intra`  
**User:** `abedin`  
**Project:** `/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2`
