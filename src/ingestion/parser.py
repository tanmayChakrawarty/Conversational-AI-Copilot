import re
from typing import Dict, Generator


def parse_transcript(file_path: str, call_id: str) -> Generator[Dict, None, None]:
    """
    Parses a transcript file with timestamps and complex speaker roles.

    This function reads a transcript file line by line and yields a dictionary
    for each continuous speech segment from a single speaker. It is designed to
    handle formats like '[MM:SS] Speaker Role (Name): Text...' which may span
    multiple lines.

    Args:
        file_path: The full path to the transcript .txt file.
        call_id: A unique identifier for the call, typically the filename.

    Yields:
        A generator of dictionaries, where each dictionary represents a
        segment of speech and contains the call_id, timestamp, speaker,
        and the full text of that segment.
    """
    # This regex is designed to capture the three key parts of a line that starts a new turn:
    # 1. (\\[\\d{2}:\\d{2}\\]): Captures a timestamp like `[00:49]`.
    # 2. (.*?): Non-greedily captures the speaker's name/role up to the colon.
    # 3. (.*): Captures the rest of the line, which is the beginning of their speech.
    speaker_pattern = re.compile(r"^(\[\d{2}:\d{2}\])\s+(.*?):\s+(.*)")

    current_speaker = None
    current_timestamp = None
    current_speech = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # We only strip trailing whitespace to preserve indentation in multi-line text
            line = line.rstrip()

            match = speaker_pattern.match(line)
            if match:
                # This line indicates a new speaker is talking.

                # First, if we have a pending speech segment from the previous speaker,
                # we yield it before processing the new one.
                if current_speaker and current_speech:
                    yield {
                        'call_id': call_id,
                        'timestamp': current_timestamp,
                        'speaker': current_speaker,
                        'text': '\n'.join(current_speech)  # Join with newlines to keep formatting
                    }

                # Now, we reset our variables for the new speaker's segment.
                timestamp, speaker, text = match.groups()
                current_timestamp = timestamp
                current_speaker = speaker
                current_speech = [text]  # Start a new list for this speaker's text

            elif line and current_speaker:
                # If the line doesn't match the pattern but is not empty, it's a
                # continuation of the current speaker's message (e.g., bullet points).
                current_speech.append(line.strip())  # Strip leading space for continuation lines

    # After the loop finishes, we need to yield the very last speech segment captured.
    if current_speaker and current_speech:
        yield {
            'call_id': call_id,
            'timestamp': current_timestamp,
            'speaker': current_speaker,
            'text': '\n'.join(current_speech)
        }