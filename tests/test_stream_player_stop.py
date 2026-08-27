import threading

from RealtimeTTS.stream_player import StreamPlayer


class _EmptyBuffer:
    def empty(self):
        return True


class _StoppingBufferManager:
    def __init__(self, player):
        self.player = player
        self.audio_buffer = _EmptyBuffer()
        self.cleared = False

    def get_from_buffer(self):
        self.player.playback_active = False
        return False, None

    def clear_buffer(self):
        self.cleared = True


class _FakeAudioStream:
    def __init__(self):
        self.closed = False

    def close_stream(self):
        self.closed = True


def test_process_buffer_marks_playback_inactive_on_exit():
    player = object.__new__(StreamPlayer)
    player.playback_active = True
    player.immediate_stop = threading.Event()
    player.on_playback_stop = None
    player.buffer_manager = _StoppingBufferManager(player)

    player._process_buffer()

    assert player.playback_active is False


def test_immediate_stop_joins_and_cleans_up_playback_thread():
    player = object.__new__(StreamPlayer)
    player.playback_active = True
    player.immediate_stop = threading.Event()
    player.audio_stream = _FakeAudioStream()
    player.buffer_manager = _StoppingBufferManager(player)

    def playback_worker():
        player.immediate_stop.wait(timeout=1.0)
        player.playback_active = False

    player.playback_thread = threading.Thread(target=playback_worker)
    player.playback_thread.start()

    player.stop(immediate=True)

    assert player.playback_active is False
    assert player.audio_stream.closed is True
    assert player.buffer_manager.cleared is True
    assert player.playback_thread is None
    assert player.immediate_stop.is_set() is False
