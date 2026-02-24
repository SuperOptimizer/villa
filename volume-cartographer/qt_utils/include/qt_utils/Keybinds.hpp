#pragma once

#include <QKeySequence>
#include <QString>
#include <QVector>

namespace qt_utils
{

/// Describes the type of key sequence: a text string ("Ctrl+S") or a
/// platform-standard key (QKeySequence::Save).
enum class ShortcutKind {
    Text,     ///< sequenceText is a portable key string
    Standard  ///< standardKey is a QKeySequence::StandardKey enum value
};

/// Definition of a keyboard shortcut (typically bound to a QAction / QShortcut).
struct ShortcutDef {
    const char* id;            ///< Unique identifier (e.g. "open_file")
    const char* section;       ///< Logical grouping for help text (e.g. "File")
    const char* description;   ///< Human-readable description
    ShortcutKind kind;         ///< How the key sequence is specified
    const char* sequenceText;  ///< Key string (when kind == Text), may be nullptr
    QKeySequence::StandardKey standardKey; ///< Standard key (when kind == Standard)
};

/// Definition of a raw key-press binding (typically matched in keyPressEvent).
struct KeyPressDef {
    const char* id;            ///< Unique identifier
    const char* section;       ///< Logical grouping for help text
    const char* description;   ///< Human-readable description
    Qt::Key key;               ///< The key
    Qt::KeyboardModifiers modifiers; ///< Required modifier keys
    bool requireNoAutoRepeat;  ///< If true, only fires on actual key-down, not auto-repeat
};

/// Resolve a ShortcutDef to a QKeySequence.
[[nodiscard]] auto sequenceFor(const ShortcutDef& def) -> QKeySequence;

/// Resolve a KeyPressDef to a QKeySequence (for display purposes).
[[nodiscard]] auto sequenceFor(const KeyPressDef& def) -> QKeySequence;

/// Format a QKeySequence as a human-readable native string.
[[nodiscard]] auto formatSequence(const QKeySequence& sequence) -> QString;

/// A keybind registry that collects shortcut and key-press definitions and
/// can generate formatted help text grouped by section.
class KeybindRegistry
{
public:
    KeybindRegistry() = default;

    /// Register a shortcut definition. The pointed-to ShortcutDef must outlive
    /// the registry (typically a static/extern const).
    void registerShortcut(const ShortcutDef& def);

    /// Register a key-press definition.
    void registerKeyPress(const KeyPressDef& def);

    /// Register a literal help-text entry (e.g. "Mouse Wheel: Zoom in/out").
    void registerLiteral(
        const char* section,
        const char* description,
        const char* keyText);

    /// Look up a ShortcutDef by its id. Returns nullptr if not found.
    [[nodiscard]] auto findShortcut(const char* id) const -> const ShortcutDef*;

    /// Look up a KeyPressDef by its id. Returns nullptr if not found.
    [[nodiscard]] auto findKeyPress(const char* id) const -> const KeyPressDef*;

    /// Build a formatted help-text string with all registered bindings,
    /// grouped by section.
    [[nodiscard]] auto buildHelpText() const -> QString;

private:
    enum class EntryKind { Shortcut, KeyPress, Literal };

    struct HelpEntry {
        const char* section;
        const char* description;
        EntryKind kind;
        const ShortcutDef* shortcut{nullptr};
        const KeyPressDef* keypress{nullptr};
        const char* literal{nullptr};
    };

    QVector<HelpEntry> entries_;
};

}  // namespace qt_utils
