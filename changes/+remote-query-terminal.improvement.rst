Changed the remote query terminal API to use ``RemoteQuery.submit()`` for
explicit async submission and ``RemoteQuery.fetch()`` for the one-shot
submit-wait-download-open flow. ``RemoteQuery.get()`` was removed.
