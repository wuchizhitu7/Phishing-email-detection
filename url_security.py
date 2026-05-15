"""
URL 安全检测模块 - 本地规则评级，不依赖外部 API
"""
from urllib.parse import urlparse
import re

# 短网址服务
SHORTENERS = {
    'bit.ly', 't.cn', 'tinyurl.com', 'ow.ly', 'is.gd', 'buff.ly',
    'goo.gl', 'short.url', 'rb.gy', 'cutt.ly', 'rebrand.ly',
    'shorte.st', 'bc.vc', 'v.gd', 'x.co', 'lc.chat',
}

# 钓鱼常用免费/可疑顶级域
SUSPICIOUS_TLDS = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.date'}

# 常见被仿冒品牌
BRANDS = {
    'apple', 'google', 'microsoft', 'amazon', 'paypal', 'facebook',
    'netflix', 'instagram', 'twitter', 'linkedin', 'dropbox',
    'alibaba', 'taobao', 'wechat', '微信', 'qq', 'alipay',
    '支付宝', 'jd', '京东', 'baidu', '携程', '顺丰', 'sf-express',
    'icbc', '中国银行', 'ccb', '招商银行', '农业银行', '建设银行',
}

# 钓鱼常用词
PHISH_TERMS = {
    'verify', 'secure', 'login', 'signin', 'account', 'update',
    'password', 'banking', 'confirm', 'safety', 'security',
    '验证', '登录', '安全', '账号', '账户', '密码', '确认', '银行',
}


def analyze_url(url):
    """分析单个 URL，返回风险字典"""
    result = {
        'url': url,
        'risk_level': 'safe',  # safe / low / medium / high
        'risk_score': 0,       # 0-10
        'reasons': [],
    }

    if not url or not isinstance(url, str) or '.' not in url:
        return result

    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    domain = hostname.lower()

    # 1. HTTP 非加密
    if parsed.scheme == 'http':
        result['risk_score'] += 1
        result['reasons'].append('HTTP 非加密')

    # 2. IP 直连
    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname):
        result['risk_score'] += 3
        result['reasons'].append('IP 直连')

    # 3. 短网址
    main_domain = '.'.join(domain.split('.')[-3:]) if len(domain.split('.')) > 2 else domain
    for s in SHORTENERS:
        if s in domain:
            result['risk_score'] += 3
            result['reasons'].append(f'短网址服务 ({s})')
            break

    # 4. 可疑顶级域
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            result['risk_score'] += 2
            result['reasons'].append(f'可疑顶级域 ({tld})')
            break

    # 5. 仿冒品牌
    for brand in BRANDS:
        if brand in domain.lower():
            # 检查是否有连字符（正规品牌不用连字符分割）
            if '-' in domain:
                result['risk_score'] += 3
                result['reasons'].append(f'疑似仿冒 {brand}（含连字符）')
            elif not domain.endswith(f'{brand}.com') and not domain.endswith(f'{brand}.cn'):
                result['risk_score'] += 2
                result['reasons'].append(f'疑似仿冒 {brand}（非官方域名）')
            break

    # 6. 连字符过多（大于2个）
    hyphens = domain.count('-')
    if hyphens > 2:
        result['risk_score'] += 1
        result['reasons'].append(f'域名连字符过多 ({hyphens}个)')

    # 7. 钓鱼关键词在域名中
    for term in PHISH_TERMS:
        if term in domain.lower():
            result['risk_score'] += 1
            result['reasons'].append(f'域名含敏感词 ({term})')
            break

    # 8. 超长子域名（>4 级）
    sub_levels = len(domain.split('.'))
    if sub_levels > 4:
        result['risk_score'] += 1
        result['reasons'].append(f'子域名层级过深 ({sub_levels}级)')

    # 9. URL 含 @ 符号
    if '@' in url:
        result['risk_score'] += 4
        result['reasons'].append('URL 含 @ 符号（重定向攻击）')

    # 10. 端口号异常
    if parsed.port and parsed.port not in (80, 443, 8080, 8443):
        result['risk_score'] += 1
        result['reasons'].append(f'异常端口 ({parsed.port})')

    # 风险等级
    score = result['risk_score']
    if score == 0:
        result['risk_level'] = 'safe'
    elif score <= 1:
        result['risk_level'] = 'low'
    elif score <= 3:
        result['risk_level'] = 'medium'
    else:
        result['risk_level'] = 'high'

    return result


def analyze_urls(urls):
    """批量分析 URL 列表"""
    if not urls:
        return []
    return [analyze_url(u) for u in urls]


def format_url_report(results):
    """格式化 URL 安全报告为可显示文本"""
    if not results:
        return '未提取到链接'

    lines = []
    for r in results:
        icon = {'safe': '✅', 'low': '🟢', 'medium': '🟠', 'high': '🔴'}[r['risk_level']]
        lines.append(f"{icon} [{r['risk_level']}] {r['url'][:80]}")
        for reason in r['reasons']:
            lines.append(f"   └ {reason}")
        lines.append('')
    return '\n'.join(lines)


if __name__ == "__main__":
    test_urls = [
        "https://appleid-verify.safety-check.cn/verify?id=xxx",
        "http://192.168.1.1/login.php",
        "https://bit.ly/abc123",
        "https://www.apple.com/verify",
        "https://secure-banking-update.xyz/login",
        "https://mail.google.com",
    ]
    for r in analyze_urls(test_urls):
        print(f"[{r['risk_level']}] score={r['risk_score']} {r['url']}")
        for reason in r['reasons']:
            print(f"  -> {reason}")
        print()
