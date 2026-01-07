from email_validator import validate_email

try:
    v = validate_email('user@nonexistent-domain-xyz-abc.com')
except Exception as e:
    print('error', type(e), e)
else:
    print(type(v))
    print('dir:', dir(v))
    for name in ('email','normalized','domain','local_part'):
        if hasattr(v, name):
            print(name, ':', getattr(v, name))
