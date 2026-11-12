"""Manual GPU fabric link controls on the authoritative Runtime thread."""


def apply_device_command(engine, command):
    net, identity, action = engine.net, command.get("target_id"), command.get("action")
    if not net.start_s <= engine.now < engine.end:
        raise ValueError("只能在运行时域内修改 GPU 链路")
    link = next((link for link in net.topology.links if link.id == identity), None)
    if (
        link is None
        or net.topology.devices[link.source].satellite
        != net.topology.devices[link.target].satellite
    ):
        raise ValueError("请选择卫星内部的 GPU / 交换互联链路")
    if not link.enabled:
        raise ValueError("此链路在配置中未启用，请修改配置后重新运行")
    operations = {
        "device_disconnect": (net.disabled_device_links, True, "GPU 链路已断开"),
        "device_connect": (net.disabled_device_links, False, "GPU 链路已重连"),
        "device_link_fail": (net.failed_device_links, True, "GPU 链路故障已生效"),
        "device_link_restore": (net.failed_device_links, False, "GPU 链路故障已解除"),
    }
    if action not in operations:
        raise ValueError("不支持的 GPU 链路操作")
    collection, adding, message = operations[action]
    if (identity in collection) == adding:
        raise ValueError("该链路已处于目标状态")
    if adding:
        collection.add(identity)
    else:
        collection.remove(identity)
    if net.physical is not None:
        net.physical.revision += 1
        net._project()
    else:
        net.revision += 1
    engine.schedule_reallocation()
    engine.interrupt_unavailable("manual_gpu_link_changed")
    engine.wake_retries()
    event = dict(
        kind="topology_command",
        time_s=engine.now,
        action=action,
        target_id=identity,
        source="manual",
        message=message,
        topology_revision=net.revision,
        scope="gpu_fabric",
    )
    engine.log.append(event)
    return event
