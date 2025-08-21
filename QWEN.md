# Qwen Development Guidelines

This document outlines the key principles and practices to follow when working on this project.

## Core Principles

1. **Server-First Development**: 
   - Never test on your local machine. 
   - Assume the code will always run on a server or remote machine.
   - This ensures consistency and avoids environment-specific issues.

2. **Code Structure**:
   - Keep actual Python code in a single file as much as possible.
   - This simplifies deployment and reduces complexity.

3. **Configuration Management**:
   - Keep all options in configuration files.
   - Create new configuration versions instead of modifying code for small changes.
   - This allows for easy experimentation and rollbacks.

4. **Documentation**:
   - Keep the README up-to-date with the latest changes.
   - Document any new features, configurations, or usage instructions.

## Additional Guidelines

5. **Output Format**:
   - Default output format should be CSV.
   - All JSON configuration files should specify CSV as the default output format.

6. **Documentation Search**:
   - Search relevant documentation in Context7 MCP when needed.

7. **Version Control**:
   - Always push changes to git after implementing features or fixes.
   - Use `git add . && git commit -m "Brief description of changes" && git push` to commit and push changes.

By following these guidelines, we ensure a consistent and maintainable codebase that's optimized for server environments.