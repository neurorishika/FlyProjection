
def validate_ip_address(ip):
    """
    Validate an IPv4 address.

    Parameters
    ----------
    ip : str
        The IP address to validate.

    Returns
    -------
    bool
        True if the IP address is valid, False otherwise.
    """
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if not 0 <= int(part) <= 255:
            return False
    return True

def validate_port(port):
    """
    Validate a port number.

    Parameters
    ----------
    port : str
        The port number to validate.

    Returns
    -------
    bool
        True if the port number is valid, False otherwise.
    """
    if not port.isdigit():
        return False
    if not 0 <= int(port) <= 65535:
        return False
    return True

def ping_ip_address(ip):
    """
    Ping an IP address to check if it is reachable.

    Parameters
    ----------
    ip : str
        The IP address to ping.

    Returns
    -------
    bool
        True if the IP address is reachable, False otherwise.
    """
    try:
        response = os.system(f"ping -c 1 {ip}")
        if response == 0:
            return True
    except Exception as e:
        print(f"Error pinging IP address: {e}")
    return False