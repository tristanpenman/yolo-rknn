#!/usr/bin/env bash
set -euo pipefail

# Create a user and group matching the host's UID/GID, so files written to
# bind-mounted volumes are owned by the host user rather than root.
#
# The container starts as root (the default) so we have permission to create
# the user/group here, then we drop to that user with gosu before running the
# requested command.

USER_ID="${HOST_UID:-1000}"
GROUP_ID="${HOST_GID:-1000}"
USERNAME=docker

# Create the group, reusing any existing group that already has this GID.
if getent group "$GROUP_ID" >/dev/null; then
  GROUP_NAME="$(getent group "$GROUP_ID" | cut -d: -f1)"
else
  GROUP_NAME="$USERNAME"
  groupadd -g "$GROUP_ID" "$GROUP_NAME"
fi

# Create the user, reusing any existing user that already has this UID.
if getent passwd "$USER_ID" >/dev/null; then
  USERNAME="$(getent passwd "$USER_ID" | cut -d: -f1)"
else
  useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash "$USERNAME" 2> /dev/null
fi

export HOME="/home/$USERNAME"

# Default to an interactive shell when no command is given.
if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec gosu "$USER_ID:$GROUP_ID" "$@"
