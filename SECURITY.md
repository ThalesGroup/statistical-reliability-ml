# Statistical Reliability ML security policy

## Goods practices to follow

:warning:**You must never store credentials information into source code or config file in a GitHub repository**
- Block sensitive data being pushed to GitHub by git-secrets or its likes as a git pre-commit hook
- Audit for slipped secrets with dedicated tools
- Use environment variables for secrets in CI/CD (e.g. GitHub Secrets) and secret managers in production

## Reporting a Vulnerability

Vulnerabilities can be reported through the github [issues](https://github.com/ThalesGroup/statistical-reliability-ml/issues tracker). Once accepted, the vulnerability will be treated as soon as possible.

## Disclosure policy

## Security Update policy

## Security related configuration

## Known security gaps & future enhancements
