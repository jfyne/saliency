# Development Workflow

## Testing Approach

**Test Alongside Development**
- Write pytest unit tests as features are developed
- Validate models on advertising-specific datasets
- Manual testing for model quality assessment

Current status: pytest configured in pyproject.toml, test suite to be developed

## Commit Message Style

**Conventional Commits with Scope**

Format: `<type>(<scope>): <description>`

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `refactor`: Code restructuring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

Example: `feat(metal): add Apple Silicon GPU support via tensorflow-metal`

## Code Review Process

**Self-Review Workflow**
- Solo developer workflow with self-review
- Use PRs for major architectural changes (optional)
- Document significant changes in docs/research/ or docs/plans/

## Pre-Merge Requirements

- [ ] Ruff linting passes (`ruff check .`)
- [ ] Manual model validation on representative advertising datasets
- [ ] Cross-platform compatibility verified when touching device-specific code
- [ ] Weight format compatibility documented if data format changes

## Documentation Standards

- Research documents: `docs/research/YYYY-MM-DD-topic.md`
- Implementation plans: `docs/plans/topic.md`
- Knowledge base: `docs/knowledge/topic.md`
- Update README.md for user-facing changes
