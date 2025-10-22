# SGLang项目构建和开发工具Makefile
# 这个Makefile提供了SGLang项目的常用开发任务，包括代码格式化、版本更新等

.PHONY: check-deps install-deps format update help  # 声明伪目标，确保这些目标总是被执行

# 显示所有可用目标的帮助信息
help:
	@echo "Available targets:"  # 显示可用目标标题
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'  # 提取并格式化显示所有目标及其描述

# 检查并安装Python代码格式化所需的依赖
check-deps: ## Check and install required Python formatting dependencies
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && pip install isort)  # 检查isort是否安装，未安装则安装
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && pip install black)  # 检查black是否安装，未安装则安装

# 安装Python代码格式化工具
install-deps: ## Install Python formatting tools (isort and black)
	pip install isort black  # 安装isort（导入排序）和black（代码格式化）工具

# 使用isort和black格式化修改的Python文件
format: check-deps ## Format modified Python files using isort and black
	@echo "Formatting modified Python files..."  # 显示格式化开始信息
	git diff --name-only --diff-filter=M | grep '\.py$$' | xargs -I {} sh -c 'isort {} && black {}'  # 获取修改的Python文件并格式化

# 定义需要更新版本号的文件列表
FILES_TO_UPDATE = docker/Dockerfile.rocm \  # ROCm Docker文件
                 python/pyproject.toml \  # Python项目配置
                 python/sglang/version.py \  # 版本定义文件
                 docs/references/setup_github_runner.md \  # GitHub Runner设置文档
                 docs/start/install.md \  # 安装文档
				 benchmark/deepseek_v3/README.md  # DeepSeek V3基准测试文档

# 更新项目文件中所有版本号
update: ## Update version numbers across project files. Usage: make update <new_version>
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \  # 检查是否提供了新版本号
		echo "Version required. Usage: make update <new_version>"; \  # 显示使用说明
		exit 1; \  # 退出并返回错误码
	fi
	@OLD_VERSION=$$(grep "version" python/sglang/version.py | cut -d '"' -f2); \  # 从版本文件中提取当前版本号
	NEW_VERSION=$(filter-out $@,$(MAKECMDGOALS)); \  # 获取新版本号
	echo "Updating version from $$OLD_VERSION to $$NEW_VERSION"; \  # 显示版本更新信息
	for file in $(FILES_TO_UPDATE); do \  # 遍历需要更新的文件列表
		if [ "$(shell uname)" = "Darwin" ]; then \  # 检查是否为macOS系统
			sed -i '' -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \  # macOS版本的sed命令
		else \
			sed -i -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \  # Linux版本的sed命令
		fi \
	done; \  # 结束文件遍历循环
	echo "Version update complete"  # 显示版本更新完成信息

# 默认目标，用于处理未定义的目标
%:
	@:  # 空操作，忽略未定义的目标
