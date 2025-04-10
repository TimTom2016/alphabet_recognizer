train:
    cargo run --release --package train --features wgpu

build_web:
    cd web && ./build-for-web.sh wgpu

run_web:
    cd web && ./run-server.sh
